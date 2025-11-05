#ifndef RT_POOL_H
#define RT_POOL_H

#include <nexus-api.h>

#include <algorithm>
#include <list>
#include <vector>
#include <iostream>

namespace nxs {
namespace rt {

/**
 * Simple object pool using std::vector
 * Objects are constructed on-demand and can be reused
 */
template <typename T, size_t initial_capacity = 256>
class Pool {
 private:
  std::vector<T> objects;
  std::list<nxs_int> available_indices;
  
 public:
  Pool() {
    objects.reserve(initial_capacity);
  }
  
  ~Pool() { 
    std::cout << "Pool<" << typeid(T).name() << "> destroyed: "
              << objects.size() << " objects, " 
              << get_in_use_count() << " still in use" << std::endl;
    clear(); 
  }

  void print_stats(const std::string& label = "") const {
    std::cout << label << " Pool: "
              << get_in_use_count() << " in use, "
              << available_indices.size() << " available, "
              << objects.size() << " total" << std::endl;
  }

  template <typename... Args>
  nxs_int acquire(Args&&... args) {
    nxs_int index;
    
    if (!available_indices.empty()) {
      // Reuse available slot
      index = available_indices.front();
      available_indices.pop_front();
      // Reconstruct in place
      objects[index].~T();
      new (&objects[index]) T(std::forward<Args>(args)...);
    } else {
      // Create new object
      index = static_cast<nxs_int>(objects.size());
      objects.emplace_back(std::forward<Args>(args)...);
    }
    
    // Debug: print stats every 100 allocations
    static int count = 0;
    if (++count % 100 == 0) {
      print_stats("After " + std::to_string(count) + " acquires");
    }
    
    return index;
  }

  template <typename... Args>
  T* get_new(Args&&... args) {
    nxs_int index = acquire(std::forward<Args>(args)...);
    return get(index);
  }

  void release(T* obj) {
    if (!obj || objects.empty()) return;
    
    if (obj >= &objects[0] && obj < &objects[0] + objects.size()) {
      nxs_int index = static_cast<nxs_int>(obj - &objects[0]);
      release(index);
    }
  }
  
  void release(nxs_int index) {
    if (index < 0 || index >= static_cast<nxs_int>(objects.size())) return;
    
    // Add to available list if not already there
    if (std::find(available_indices.begin(), available_indices.end(), index) 
        == available_indices.end()) {
      available_indices.push_back(index);
    }
  }

  T* get(nxs_int index) {
    if (index < 0 || index >= static_cast<nxs_int>(objects.size())) {
      return nullptr;
    }
    return &objects[index];
  }

  std::pair<size_t, size_t> get_stats() const {
    return {available_indices.size(), objects.size()};
  }

  void clear() {
    objects.clear();
    available_indices.clear();
  }

  void reserve(size_t capacity) {
    objects.reserve(capacity);
  }

  size_t capacity() const { 
    return objects.size(); 
  }

  size_t get_in_use_count() const {
    return objects.size() - available_indices.size();
  }

  bool owns_object(const T* obj) const {
    if (!obj || objects.empty()) return false;
    return obj >= &objects[0] && obj < &objects[0] + objects.size();
  }
};

}  // namespace rt
}  // namespace nxs

#endif  // RT_POOL_H