#pragma once

#include <panel_detector_param/panel_detector_param.hpp>
#include <vector>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <opencv2/opencv.hpp>

namespace panel_detector_base
{
class Detector
{
public:
  virtual void init(const panel_detector_parameters::ParamListener & param_listener) = 0;
  virtual vision_msgs::msg::Detection2DArray detect(const cv::Mat & image) = 0;
  virtual ~Detector() {}

protected:
  Detector() {}
};
}
