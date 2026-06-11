#pragma once

#include <uv.h>
#include <atomic>
#include <deque>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <chrono>

#include "runtime/request.h"
#include "runtime/response.h"
#include "runtime/scheduler.h"
#include "constants.h"
namespace mini_llm::runtime {
class ServerContext;

struct ClientConnection {
    uv_tcp_t handle{};
    ServerContext* server = nullptr;

    std::string read_buffer;
    std::deque<std::string> write_queue;

    bool writing = false;
    bool closed = false;
};

class ServerContext {
private:
    struct WriteRequest {
        uv_write_t req{};
        uv_buf_t buf{};
        ClientConnection* client = nullptr;
        std::string data;
    };

private:
    uv_loop_t* loop_ = nullptr;
    uint64_t next_request_id = 1;
    uv_tcp_t server_handle_{};
    uv_async_t response_async_{};

    MutexQueue<std::unique_ptr<Request>> request_queue_;
    MutexQueue<Response> response_queue_;

    std::atomic<bool> running_{false};
    std::unordered_map<uint64_t, ClientConnection*> req_to_client_;
    std::unique_ptr<Scheduler> scheduler_;
public:
    ServerContext();
    ~ServerContext();

    ServerContext(const ServerContext&) = delete;
    ServerContext& operator=(const ServerContext&) = delete;

    bool start(const char* host = "0.0.0.0", int port = mini_llm::constants::DEFAULT_PORT);
    void run();
    void stop();

    void submit_request(std::unique_ptr<Request> req);
    void submit_response(Response resp);

private:
    static void alloc_buffer_static(
        uv_handle_t* handle,
        std::size_t suggested_size,
        uv_buf_t* buf
    );

    static void on_new_connection_static(
        uv_stream_t* server,
        int status
    );

    static void on_read_static(
        uv_stream_t* stream,
        ssize_t nread,
        const uv_buf_t* buf
    );

    static void on_client_closed_static(
        uv_handle_t* handle
    );

    static void on_response_async_static(
        uv_async_t* handle
    );

    static void on_write_static(
        uv_write_t* req,
        int status
    );

private:
    void on_new_connection(uv_stream_t* server, int status);

    void on_read(
        ClientConnection* client,
        ssize_t nread,
        const uv_buf_t* buf
    );

    void on_response_async();

    void enqueue_write(ClientConnection* client, std::string data);
    void start_write(ClientConnection* client);

};
}