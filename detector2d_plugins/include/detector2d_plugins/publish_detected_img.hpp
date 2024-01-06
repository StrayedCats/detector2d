#pragma once

#include <cmath>
#include <chrono>

#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include "bboxes_ex_msgs/msg/bounding_box.hpp"
#include "bboxes_ex_msgs/msg/bounding_boxes.hpp"

#include <pluginlib/class_loader.hpp>

#include "yolox_cpp/yolox.hpp"
#include "yolox_cpp/utils.hpp"
#include "yolox_param/yolox_param.hpp"

#include <detector2d_base/detector2d_base.hpp>
#include <detector2d_param/detector2d_param.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

namespace detector2d_plugins
{

    class publish_detected_img : public detector2d_base::Detector
    {
    public:
        void init(const detector2d_parameters::ParamListener &) override;
        vision_msgs::msg::Detection2DArray detect(const cv::Mat &) override;

    protected:
        std::shared_ptr<yolox_parameters::ParamListener> yolox_params_;
        yolox_parameters::Params params_;

    private:
        std::unique_ptr<yolox_cpp::AbcYoloX> yolox_;
        std::vector<std::string> class_names_;

        bboxes_ex_msgs::msg::BoundingBoxes objects_to_bboxes(cv::Mat, std::vector<yolox_cpp::Object>, std_msgs::msg::Header);
        vision_msgs::msg::Detection2DArray objects_to_Detection2DArray(cv::Mat frame, std::vector<yolox_cpp::Object> objects, std_msgs::msg::Header header);
    };

} // namespace detector2d_plugins
