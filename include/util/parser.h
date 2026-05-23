#include <vector>
#include <string>
#include <stdexcept>
#include <sstream>

struct InferRequest {
  std::vector<int> input_ids;
  int max_tokens = 50;
};

InferRequest parse_request(const std::string& body) {
  InferRequest req;

  // input_ids 배열 추출
  size_t arr_start = body.find('[');
  size_t arr_end   = body.find(']');
  if (arr_start == std::string::npos || arr_end == std::string::npos)
      throw std::runtime_error("input_ids not found");

  std::string arr = body.substr(arr_start + 1, arr_end - arr_start - 1);
  std::stringstream ss(arr);
  std::string token;
  while (std::getline(ss, token, ',')) {
      req.input_ids.push_back(std::stoi(token));
  }

  // max_tokens 추출
  size_t mt_pos = body.find("max_tokens");
  if (mt_pos != std::string::npos) {
      size_t colon = body.find(':', mt_pos);
      req.max_tokens = std::stoi(body.substr(colon + 1));
  }

  return req;
}

// output_ids 직렬화
std::string make_response(const std::vector<int>& output_ids) {
  std::string s = "{\"output_ids\": [";
  for (int i = 0; i < (int)output_ids.size(); ++i) {
      if (i > 0) s += ", ";
      s += std::to_string(output_ids[i]);
  }
  s += "]}";
  return s;
}
