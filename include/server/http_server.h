#pragma once
#include <mutex>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include <queue>
struct HttpRequest {
  std::string method;
  std::string path;
  std::string body;
  bool complete = false;
};

struct HttpResponse {
  int status = 200;
  std::string body;
};

struct CompletedResult {
  int client_fd;
  std::string response_body;
};


class HttpServer {
public:
  using Handler = std::function<HttpResponse(const HttpRequest&)>;

  HttpServer(int port, Handler handler, int max_events = 64);
  ~HttpServer();

  void run();
  void stop();

private:
  void setup_listen_socket();
  void event_loop();
  void setup_notify_fd();

  void on_accept();
  void on_readable(int fd);
  void on_writable(int fd);
  void on_notify();
  void close_conn(int fd);

  void try_dispatch(int fd);           
  void enqueue_response(int fd, const HttpResponse& resp); 

  void notify_complete(int client_fd, std::string body);
              

  int     port_;
  int     listen_fd_ = -1;
  int     epoll_fd_  = -1;
  int     notify_fd_ = -1;
  int     max_events_;
  bool    running_   = false;

  Handler handler_;

  std::unordered_map<int, std::string> read_buf_;
  std::unordered_map<int, std::string> write_buf_;

  std::mutex completed_mutex_;
  std::queue<CompletedResult> completed_;
};


