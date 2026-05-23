#pragma once

#include <queue>
#include <mutex>
#include <optional>
#include <vector>


template<typename T>
class MpscQueue {
public:
    void push(T item);

    std::optional<T> try_pop();

    void drain(std::vector<T>& out);
    bool   empty() const;
    size_t size()  const;

private:
    std::queue<T>      queue_;
    mutable std::mutex mutex_;
};


template<typename T>
void MpscQueue<T>::push(T item) {
    // TODO
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(std::move(item));
}

template<typename T>
std::optional<T> MpscQueue<T>::try_pop() {
    // TODO
    std::lock_guard<std::mutex> lock(mutex_);
    if(queue_.size() > 0) {
        T ret = std::move(queue_.front());
        queue_.pop();
        return ret;
    }
    return std::nullopt;
}

template<typename T>
void MpscQueue<T>::drain(std::vector<T>& out) {
    // TODO
    std::lock_guard<std::mutex> lock(mutex_);
    while(!queue_.empty()) {
        //T cur = std::move(queue_.front());
        //queue_.pop();
        //out.push_back(cur);
	out.push_back(std::move(queue_.front()));
        queue_.pop();
    }
}

template<typename T>
bool MpscQueue<T>::empty() const {
    // TODO
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

template<typename T>
size_t MpscQueue<T>::size() const {
    // TODO
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}
