#include "detector2d_plugins/publish_detected_img.hpp"

// for TensorRT
#define ENABLE_TENSORRT
namespace detector2d_plugins
{
  void publish_detected_img::init(const detector2d_parameters::ParamListener &param_listener_)
  {
    // debug
    std::cout << "publish_detected_img::init" << std::endl;

    if (this->params_.imshow_isshow)
    {
      cv::namedWindow("yolox", cv::WINDOW_AUTOSIZE);
    }

    if (this->params_.class_labels_path != "")
    {
      this->class_names_ = yolox_cpp::utils::read_class_labels_file(this->params_.class_labels_path);
    }
    else
    {
      this->class_names_ = yolox_cpp::COCO_CLASSES;
    }

    if (this->params_.model_type == "tensorrt")
    {
#ifdef ENABLE_TENSORRT
      this->yolox_ = std::make_unique<yolox_cpp::YoloXTensorRT>(
          this->params_.model_path, this->params_.tensorrt_device,
          this->params_.nms, this->params_.conf, this->params_.model_version,
          this->params_.num_classes, this->params_.p6);
#else
      rclcpp::shutdown();
#endif
    }
    else if (this->params_.model_type == "openvino")
    {
#ifdef ENABLE_OPENVINO
      this->yolox_ = std::make_unique<yolox_cpp::YoloXOpenVINO>(
          this->params_.model_path, this->params_.openvino_device,
          this->params_.nms, this->params_.conf, this->params_.model_version,
          this->params_.num_classes, this->params_.p6);
#else
      rclcpp::shutdown();
#endif
    }
    else if (this->params_.model_type == "onnxruntime")
    {
#ifdef ENABLE_ONNXRUNTIME
      this->yolox_ = std::make_unique<yolox_cpp::YoloXONNXRuntime>(
          this->params_.model_path,
          this->params_.onnxruntime_intra_op_num_threads,
          this->params_.onnxruntime_inter_op_num_threads,
          this->params_.onnxruntime_use_cuda, this->params_.onnxruntime_device_id,
          this->params_.onnxruntime_use_parallel,
          this->params_.nms, this->params_.conf, this->params_.model_version,
          this->params_.num_classes, this->params_.p6);
#else
      rclcpp::shutdown();
#endif
    }
    else if (this->params_.model_type == "tflite")
    {
#ifdef ENABLE_TFLITE
      this->yolox_ = std::make_unique<yolox_cpp::YoloXTflite>(
          this->params_.model_path, this->params_.tflite_num_threads,
          this->params_.nms, this->params_.conf, this->params_.model_version,
          this->params_.num_classes, this->params_.p6, this->params_.is_nchw);
#else
      rclcpp::shutdown();
#endif
    }
    (void)param_listener_;

    // debug
    std::cout << "publish_detected_img::init end" << std::endl;
  }

  vision_msgs::msg::Detection2DArray publish_detected_img::detect(const cv::Mat &frame)
  {
    // 検出
    auto objects = this->yolox_->inference(frame);

    // fpsの計算
    // auto now = std::chrono::system_clock::now();
    // auto end = std::chrono::system_clock::now();
    // auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - now);
    // RCLCPP_INFO(this->get_logger(), "Inference: %f FPS", 1000.0f / elapsed.count());

    // 検出した物体に描画
    yolox_cpp::utils::draw_objects(frame, objects, this->class_names_);

    if (this->params_.imshow_isshow)
    {
      cv::imshow("yolox", frame);
      auto key = cv::waitKey(1);
      if (key == 27)
      {
        rclcpp::shutdown();
      }
    }

    // ボックスのデータをpublish
    std_msgs::msg::Header header;
    // stamp is undefined
    header.frame_id = "yolox";
    auto boxes = this->objects_to_Detection2DArray(frame, objects, header);
    // this->pub_bboxes_->publish(boxes);

    // 画像データのpublish
    /*
    //publishするための変数を作成
    sensor_msgs::msg::Image::SharedPtr pub_img;
    //img_msgs型に変換
    pub_img = cv_bridge::CvImage(img->header, "bgr8", frame).toImageMsg();
    //publish
    this->pub_image_.publish(pub_img);
    */
    // TODO : 変換と返り値に変更

    return boxes;
  }

  bboxes_ex_msgs::msg::BoundingBoxes publish_detected_img::objects_to_bboxes(cv::Mat frame, std::vector<yolox_cpp::Object> objects, std_msgs::msg::Header header)
  {
    bboxes_ex_msgs::msg::BoundingBoxes boxes;
    boxes.header = header;
    for (auto obj : objects)
    {
      bboxes_ex_msgs::msg::BoundingBox box;
      box.probability = obj.prob;
      box.class_id = yolox_cpp::COCO_CLASSES[obj.label];
      box.xmin = obj.rect.x;
      box.ymin = obj.rect.y;
      box.xmax = (obj.rect.x + obj.rect.width);
      box.ymax = (obj.rect.y + obj.rect.height);
      box.img_width = frame.cols;
      box.img_height = frame.rows;
      boxes.bounding_boxes.emplace_back(box);
    }
    return boxes;
  }

  vision_msgs::msg::Detection2DArray publish_detected_img::objects_to_Detection2DArray(cv::Mat frame, std::vector<yolox_cpp::Object> objects, std_msgs::msg::Header header)
  {
    vision_msgs::msg::Detection2DArray boxes;
    boxes.header = header;
    for (auto obj : objects)
    {
      vision_msgs::msg::Detection2D box;
      box.header = header;

      // hypothsis
      vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
      hypothesis.hypothesis.class_id = yolox_cpp::COCO_CLASSES[obj.label];
      hypothesis.hypothesis.score = obj.prob;
      // hypothesis.pose is undefined
      box.results.emplace_back(hypothesis);

      // box.bbox
      box.bbox.center.position.x = obj.rect.x + (obj.rect.width / 2);
      box.bbox.center.position.y = obj.rect.y + (obj.rect.height / 2);
      // box.bbox.center.theta is undefined
      box.bbox.center.theta = 0;
      box.bbox.size_x = obj.rect.width;
      box.bbox.size_y = obj.rect.height;

      // push
      boxes.detections.emplace_back(box);
    }
    return boxes;
  }
}

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(detector2d_plugins::publish_detected_img, detector2d_base::Detector)
