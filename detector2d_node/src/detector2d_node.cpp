#include <detector2d_node/detector2d_node.hpp>

namespace detector2d_node
{

  Detector2dNode::Detector2dNode(const rclcpp::NodeOptions &options)
      // TODO : use options
      : rclcpp::Node("detector2d_node", options),
        detection_loader_("detector2d_base", "detector2d_base::Detector")
  {
    this->param_listener_ = std::make_shared<detector2d_parameters::ParamListener>(
        this->get_node_parameters_interface());
    const auto params = this->param_listener_->get_params();

    try
    {
      this->detector_ = this->detection_loader_.createSharedInstance(
          params.load_target_plugin);
      this->detector_->init(*this->param_listener_);
      std::cout << "params.load_target_plugin: " << params.load_target_plugin << std::endl;
    }
    catch (pluginlib::PluginlibException &ex)
    {
      printf("The plugin failed to load for some reason. Error: %s\n", ex.what());
    }

    // debug
    std::cout << "detector2d_node::Detector2dNode before creating pub/sub" << std::endl;
    try
    {
      this->pose_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>(
          "positions", 1);
      this->image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
          "image_raw", 1, std::bind(&Detector2dNode::image_callback, this, std::placeholders::_1));
    }
    catch (rclcpp::exceptions::RCLError &ex)
    {
      printf("The publisher or subscriber failed to create for some reason. Error: %s\n", ex.what());
    }

    // debug
    std::cout << "detector2d_node::Detector2dNode created Node" << std::endl;
  }

  void Detector2dNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    vision_msgs::msg::Detection2DArray bboxes =
        this->detector_->detect(cv_bridge::toCvShare(msg, "bgr8")->image);
    for (size_t i = 0; i < bboxes.detections.size(); i++)
    {
      std::cout << "bboxes [" << i << "]: " << bboxes.detections[i].bbox.center.position.x << ", " << bboxes.detections[i].bbox.center.position.y << std::endl;
    }
    this->pose_pub_->publish(bboxes);
  }
} // namespace detector2d_node

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(detector2d_node::Detector2dNode)

// debug
int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options = rclcpp::NodeOptions();
  std::shared_ptr<detector2d_node::Detector2dNode> node = std::make_shared<detector2d_node::Detector2dNode>(options);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}