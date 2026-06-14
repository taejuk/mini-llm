#include "runtime/server.h"
#include <sstream>
#if MINI_LLM_USE_MOCK_BACKEND
#include "runtime/mock_backend.h"
#include "runtime/mock_kv_allocator.h"
#else
#include "runtime/gpt2_backend.h"
#include "runtime/real_kv_allocator.h"
#endif

namespace mini_llm::runtime {

std::vector<int> parseRequest(const std::string& line) {
    std::vector<int> tokens;
    std::istringstream iss(line);

    int token = 0;
    while (iss >> token) {
        tokens.push_back(token);
    }

    return tokens;
}

std::string tokenToString(int token) {
    return std::to_string(token);
}

ServerContext::ServerContext() {
    loop_ = uv_default_loop();
}

ServerContext::~ServerContext() {
    stop();
}

bool ServerContext::start(const char* host, int port) {
    int ret = 0;

    ret = uv_tcp_init(loop_, &server_handle_);
    if (ret != 0) {
        std::cerr << "uv_tcp_init failed: " << uv_strerror(ret) << "\n";
        return false;
    }

    server_handle_.data = this;

    sockaddr_in addr;
    ret = uv_ip4_addr(host, port, &addr);
    if (ret != 0) {
        std::cerr << "uv_ip4_addr failed: " << uv_strerror(ret) << "\n";
        return false;
    }

    ret = uv_tcp_bind(
        &server_handle_,
        reinterpret_cast<const sockaddr*>(&addr),
        0
    );

    if (ret != 0) {
        std::cerr << "uv_tcp_bind failed: " << uv_strerror(ret) << "\n";
        return false;
    }

    ret = uv_async_init(
        loop_,
        &response_async_,
        &ServerContext::on_response_async_static
    );

    if (ret != 0) {
        std::cerr << "uv_async_init failed: " << uv_strerror(ret) << "\n";
        return false;
    }

    response_async_.data = this;

    ret = uv_listen(
        reinterpret_cast<uv_stream_t*>(&server_handle_),
        mini_llm::constants::DEFAULT_BACKLOG,
        &ServerContext::on_new_connection_static
    );

    if (ret != 0) {
        std::cerr << "uv_listen failed: " << uv_strerror(ret) << "\n";
        return false;
    }

    running_.store(true);

    #if MINI_LLM_USE_MOCK_BACKEND
    backend_ = std::make_unique<MockBackend>();
    kv_allocator_ = std::make_unique<MockKvAllocator>();
    #else
    backend_ = std::make_unique<Gpt2Backend>();
    kv_allocator_ = std::make_unique<RealKvAllocator>();
    #endif

    uv_async_t* response_async = &response_async_;

    scheduler_ = std::make_unique<Scheduler>(
        request_queue_,
        response_queue_,
        response_async,
        *backend_,
        *kv_allocator_
    );

    scheduler_->start();
    std::cout << "[server] listening on " << host << ":" << port << "\n";

    return true;
}

void ServerContext::run() {
    uv_run(loop_, UV_RUN_DEFAULT);
}

void ServerContext::stop() {
    bool expected = true;

    if (!running_.compare_exchange_strong(expected, false)) {
        return;
    }

    request_queue_.close();

    // if (gpu_worker_.joinable()) {
    //     gpu_worker_.join();
    // }
    scheduler_->stop();
    uv_stop(loop_);
}

void ServerContext::submit_request(std::unique_ptr<Request> req) {
    request_queue_.push(std::move(req));
}

void ServerContext::submit_response(Response resp) {
    response_queue_.push(std::move(resp));
    uv_async_send(&response_async_);
}

void ServerContext::alloc_buffer_static(
    uv_handle_t* handle,
    std::size_t suggested_size,
    uv_buf_t* buf
) {
    (void)handle;

    char* base = new char[suggested_size];

    buf->base = base;
    buf->len = suggested_size;
}

void ServerContext::on_new_connection_static(
    uv_stream_t* server,
    int status
) {
    auto* self = static_cast<ServerContext*>(server->data);
    self->on_new_connection(server, status);
}

void ServerContext::on_read_static(
    uv_stream_t* stream,
    ssize_t nread,
    const uv_buf_t* buf
) {
    auto* client = static_cast<ClientConnection*>(stream->data);

    if (client != nullptr && client->server != nullptr) {
        client->server->on_read(client, nread, buf);
    }

    delete[] buf->base;
}

void ServerContext::on_client_closed_static(
    uv_handle_t* handle
) {
    auto* client = static_cast<ClientConnection*>(handle->data);
    delete client;
}

void ServerContext::on_response_async_static(
    uv_async_t* handle
) {
    auto* self = static_cast<ServerContext*>(handle->data);
    self->on_response_async();
}

void ServerContext::on_write_static(
    uv_write_t* req,
    int status
) {
    auto* write_req = reinterpret_cast<WriteRequest*>(req);
    ClientConnection* client = write_req->client;

    if (status < 0) {
        std::cerr << "write error: " << uv_strerror(status) << "\n";
    }

    delete write_req;

    if (client == nullptr || client->closed) {
        return;
    }

    client->writing = false;

    if (!client->write_queue.empty()) {
        client->server->start_write(client);
    }
}

void ServerContext::on_new_connection(
    uv_stream_t* server,
    int status
) {
    if (status < 0) {
        std::cerr << "new connection error: " << uv_strerror(status) << "\n";
        return;
    }

    auto* client = new ClientConnection();
    client->server = this;

    int ret = uv_tcp_init(loop_, &client->handle);
    if (ret != 0) {
        std::cerr << "uv_tcp_init client failed: " << uv_strerror(ret) << "\n";
        delete client;
        return;
    }

    client->handle.data = client;

    ret = uv_accept(
        server,
        reinterpret_cast<uv_stream_t*>(&client->handle)
    );

    if (ret != 0) {
        std::cerr << "uv_accept failed: " << uv_strerror(ret) << "\n";
        uv_close(
            reinterpret_cast<uv_handle_t*>(&client->handle),
            &ServerContext::on_client_closed_static
        );
        return;
    }

    ret = uv_read_start(
        reinterpret_cast<uv_stream_t*>(&client->handle),
        &ServerContext::alloc_buffer_static,
        &ServerContext::on_read_static
    );

    if (ret != 0) {
        std::cerr << "uv_read_start failed: " << uv_strerror(ret) << "\n";
        uv_close(
            reinterpret_cast<uv_handle_t*>(&client->handle),
            &ServerContext::on_client_closed_static
        );
        return;
    }

    std::cout << "[server] new client connected\n";
}

void ServerContext::on_read(
    ClientConnection* client,
    ssize_t nread,
    const uv_buf_t* buf
) {
    if (nread > 0) {
        client->read_buffer.append(
            buf->base,
            static_cast<std::size_t>(nread)
        );

        while (true) {
            std::size_t pos = client->read_buffer.find('\n');

            if (pos == std::string::npos) {
                break;
            }

            std::string line = client->read_buffer.substr(0, pos);
            client->read_buffer.erase(0, pos + 1);

            std::vector<int> tokens = parseRequest(line);

            if (tokens.empty()) {
                enqueue_write(client, "invalid request\n");
                continue;
            }
            req_to_client_.insert({next_request_id, client});
            auto req = std::make_unique<Request>(
                next_request_id++,
                std::move(tokens),
                8
            );

            submit_request(std::move(req));
        }

        return;
    }

    if (nread == UV_EOF) {
        std::cout << "[server] client disconnected\n";
    } else if (nread < 0) {
        std::cerr << "read error: " << uv_strerror(static_cast<int>(nread)) << "\n";
    }

    client->closed = true;

    uv_close(
        reinterpret_cast<uv_handle_t*>(&client->handle),
        &ServerContext::on_client_closed_static
    );
}

void ServerContext::on_response_async() {
    Response resp;

    while (response_queue_.try_pop(resp)) {
        auto it = req_to_client_.find(resp.request_id);

        if (it == req_to_client_.end()) {
            continue;
        }

        ClientConnection* client = it->second;

        if (client == nullptr || client->closed) {
            if (resp.finished) {
                req_to_client_.erase(it);
            }
            continue;
        }

        std::string data = tokenToString(resp.token);

        if (resp.finished) {
            data += " [DONE]";
        }

        data += "\n";

        enqueue_write(client, std::move(data));

        if (resp.finished) {
            req_to_client_.erase(it);
        }
    }
}


void ServerContext::enqueue_write(
    ClientConnection* client,
    std::string data
) {
    if (client == nullptr || client->closed) {
        return;
    }

    client->write_queue.push_back(std::move(data));

    if (!client->writing) {
        start_write(client);
    }
}

void ServerContext::start_write(ClientConnection* client) {
    if (client == nullptr || client->closed) {
        return;
    }

    if (client->writing || client->write_queue.empty()) {
        return;
    }

    std::string data = std::move(client->write_queue.front());
    client->write_queue.pop_front();

    auto* write_req = new WriteRequest();
    write_req->client = client;
    write_req->data = std::move(data);
    write_req->buf = uv_buf_init(
        write_req->data.data(),
        static_cast<unsigned int>(write_req->data.size())
    );

    client->writing = true;

    int ret = uv_write(
        &write_req->req,
        reinterpret_cast<uv_stream_t*>(&client->handle),
        &write_req->buf,
        1,
        &ServerContext::on_write_static
    );

    if (ret != 0) {
        client->writing = false;

        std::cerr << "uv_write failed: " << uv_strerror(ret) << "\n";

        delete write_req;
    }
}
}