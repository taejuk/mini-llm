#include "server/http_server.h"

#include <sys/epoll.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/eventfd.h>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <thread>

static void set_nonblocking(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  if(flags == -1) throw std::runtime_error("fcntl F_GETFL");
  if(fcntl(fd, F_SETFL, flags | O_NONBLOCK) == -1) throw std::runtime_error("fcntl F_SETFL");
}

static void epoll_add(int epfd, int fd, uint32_t events) {
  epoll_event ev{};
  ev.events = events;
  ev.data.fd = fd;
  if(epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &ev) == -1) throw std::runtime_error("epoll_ctl ADD");
}

static void epoll_mod(int epfd, int fd, uint32_t events) {
  epoll_event ev{};
  ev.events = events;
  ev.data.fd = fd;
  epoll_ctl(epfd, EPOLL_CTL_MOD, fd, &ev);
}

HttpServer::HttpServer(int port, Handler handler, int max_events)
  : port_(port), handler_(std::move(handler)), max_events_(max_events)
{
  epoll_fd_ = epoll_create1(0);
  if (epoll_fd_ == -1) throw std::runtime_error("epoll_create1");

  setup_listen_socket();
  setup_notify_fd();
}

HttpServer::~HttpServer() {
  if (listen_fd_ != -1) close(listen_fd_);
  if (epoll_fd_  != -1) close(epoll_fd_);
}

void HttpServer::setup_listen_socket() {
  listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ == -1) throw std::runtime_error("socket");

  sockaddr_in addr{};
  addr.sin_family      = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port        = htons(port_);

  if (bind(listen_fd_, (sockaddr*)&addr, sizeof(addr)) == -1)
      throw std::runtime_error("bind");

  if (listen(listen_fd_, SOMAXCONN) == -1)
      throw std::runtime_error("listen");

  set_nonblocking(listen_fd_);
  epoll_add(epoll_fd_, listen_fd_, EPOLLIN);
}

void HttpServer::setup_notify_fd() {
  notify_fd_ = eventfd(0, EFD_NONBLOCK);
  if (notify_fd_ == -1) throw std::runtime_error("eventfd");
  epoll_add(epoll_fd_, notify_fd_, EPOLLIN);
}

void HttpServer::run() {
  running_ = true;
  std::cout << "[server] listening on port " << port_ << "\n";
  event_loop();
}

void HttpServer::stop() { running_ = false; }

void HttpServer::event_loop() {
  std::vector<epoll_event> events(max_events_);

  while(running_) {
    int n = epoll_wait(epoll_fd_, events.data(), max_events_, -1);
    if(n == -1) {
      if(errno == EINTR) continue;
      break;
    }

    for(int i = 0; i < n; i++) {
      int fd = events[i].data.fd;
      auto ev = events[i].events;

      if(fd == listen_fd_) on_accept();
      else if(fd == notify_fd_) on_notify();
      else if(ev & EPOLLIN) on_readable(fd);
      else if(ev & EPOLLOUT) on_writable(fd);
      else if(ev & (EPOLLHUP | EPOLLERR)) close_conn(fd);
    }
  }
}

void HttpServer::on_accept() {
  while(true) {
    int client_fd = accept(listen_fd_, nullptr, nullptr);
    if(client_fd == -1) {
      if(errno == EAGAIN || errno == EWOULDBLOCK) break;
      perror("accept");
      break;
    }

    set_nonblocking(client_fd);
    epoll_add(epoll_fd_, client_fd, EPOLLIN);
    read_buf_[client_fd]  = "";
    write_buf_[client_fd] = "";
    std::cout << "[server] new connection fd=" << client_fd << "\n";
  }
}

void HttpServer::on_notify() {
  uint64_t val;
  read(notify_fd_, &val, sizeof(val));

  std::vector<CompletedResult> results;
  {
    std::lock_guard<std::mutex> lock(completed_mutex_);
    while(!completed_.empty()) {
      results.push_back(std::move(completed_.front()));
      completed_.pop();
    }
  }

  for(auto& r : results) {
    if(write_buf_.count(r.client_fd)) enqueue_response(r.client_fd, {200, r.response_body});
  }
}

void HttpServer::on_readable(int fd) {
  char buf[4096];
  while(true) {
    ssize_t n = read(fd, buf, sizeof(buf));
    if(n > 0) {
      read_buf_[fd].append(buf, n);
    } else if(n == 0) {
      close_conn(fd);
      return;
    } else {
      if (errno == EAGAIN || errno == EWOULDBLOCK) break;
      perror("read");
      close_conn(fd);
      return;
    }
  }

  try_dispatch(fd);
}

void HttpServer::try_dispatch(int fd) {
  auto& buf = read_buf_[fd];

  // 헤더 끝 위치 탐색
  size_t hdr_end = buf.find("\r\n\r\n");
  if (hdr_end == std::string::npos) return;  // 아직 헤더 미완성

  // ── 아주 간단한 파싱 (1줄: method + path) ──
  HttpRequest req;
  std::istringstream ss(buf.substr(0, hdr_end));
  ss >> req.method >> req.path;

  // Content-Length 추출
  size_t cl_pos = buf.find("Content-Length: ");
  size_t body_start = hdr_end + 4;
  if (cl_pos != std::string::npos) {
      size_t val_start = cl_pos + 16;
      size_t val_end   = buf.find("\r\n", val_start);
      int content_len  = std::stoi(buf.substr(val_start, val_end - val_start));

      if (buf.size() < body_start + content_len) return;  // body 미완성
      req.body = buf.substr(body_start, content_len);
  }
  req.complete = true;

  // 처리한 만큼 버퍼에서 제거
  buf.erase(0, body_start + req.body.size());

  std::thread([this, fd, req = std::move(req)]() mutable {
    HttpResponse resp = handler_(req);
    notify_complete(fd, std::move(resp.body));
  }).detach();
}

void HttpServer::notify_complete(int client_fd, std::string body) {
  {
    std::lock_guard<std::mutex> lock(completed_mutex_);
    completed_.push({client_fd, std::move(body)});
  }

  uint64_t val = 1;
  write(notify_fd_, &val, sizeof(val));
}

void HttpServer::enqueue_response(int fd, const HttpResponse& resp) {
  std::ostringstream oss;
  oss << "HTTP/1.1 " << resp.status << " OK\r\n"
      << "Content-Type: application/json\r\n"
      << "Content-Length: " << resp.body.size() << "\r\n"
      << "Connection: keep-alive\r\n"
      << "\r\n"
      << resp.body;

  write_buf_[fd] += oss.str();

  epoll_mod(epoll_fd_, fd, EPOLLIN | EPOLLOUT);
}

void HttpServer::on_writable(int fd) {
  auto& buf = write_buf_[fd];
  while(!buf.empty()) {
    ssize_t n = write(fd, buf.data(), buf.size());
    if(n > 0) buf.erase(0,n);
    else if(n == -1) {
      if(errno == EAGAIN || errno == EWOULDBLOCK) break;
      close_conn(fd);
      return;
    }
  }
  if(buf.empty()) epoll_mod(epoll_fd_, fd, EPOLLIN);
}

void HttpServer::close_conn(int fd) {
    epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, nullptr);
    close(fd);
    read_buf_.erase(fd);
    write_buf_.erase(fd);
    std::cout << "[server] closed fd=" << fd << "\n";
}
