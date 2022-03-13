from styx_msgs.msg import TrafficLight
import yaml
import os
from utils.imagesaver import ImageSaver
import tensorflow as tf
import numpy as np

class TLClassifier(object):
    def __init__(self, save_visualizations):
        #TODO load classifier
        config_string = rospy.get_param("/traffic_light_config")
        config = yaml.load(config_string)
        
        # Save every image drom the camera and annotated image with the detections
        self.save_visualizations = save_visualizations
        
        # directory in which trained models are stored
        model_dir = os.path.dirname(os.path.realpath(__file__))
        
        # load tl-detection model
        model_file_path = os.path.join(model_dir, 'tl-detection-model.pb')
        rospy.loginfo("Detection model {} loaded".format(model_file_path))
        
        # output directory for image saver
        output_dir = './data'
        is_site = config['is_site']
        
        if is_site:
            output_dir = os.path.join(output_dir, 'site')
        else:
            output_dir = os.path.join(output_dir, 'simulator')
            
        # Load TF model
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_file_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name = '')
                
        # Load label map
        label_map_file_path = os.path.join(model_dir, 'label_map.pbtxt')
        label_map = label_map_util.load_labelmap(label_map_file_path)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes = len(label_map.item),
                                                                    use_display_name = True)
                                                                    
        self.category_index = label_map_util.create_category_index(categories)
        self.image_saver = ImageSaver(self.category_index, output_dir)  

    def detect_traffic_lights(self, image):
        with self.detection_graph.as_default():
            with tf.Session(graph = self.detection_graph) as sess:
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                
                boxes, scores, classes, num_detections = sess.run([boxes, scores, classes, num_detections],
                                                                  feed_dict={image_tensor:np.expand_dims(image, axis = 0)})
                                                                  
                return boxes, scores, classes, num_detections    
    
    def predict_state(self, image):
        boxes, scores, classes, num_detections = self.detect_traffic_lights(image)
        
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)
        
        max_state = "Unknown"
        max_score = -1
        if len(scores) > 0 and scores[0] > 0.1:
            max_score = scores[0]
            max_state = self.category_index[classes[0]]['name']
            
        rospy.loginfo("Predicted sate: {} ({})".format(max_state, max_score))
        
        # save visualized image if the option is set
        if self.save_visualizations:
            self.image_saver.save_visualizations(image, boxes, classes, scores)
            
        return max_state
        
    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        #TODO implement light color prediction
        image_np = np.asarray(image, dtype='int32')
        state = self.predict_state(image_np)
        
        if state == "Red":
            return TrafficLight.RED
        elif state == "Yellow":
            return TrafficLight.YELLOW
        elif state == "Green":
            return TrafficLight.GREEN
        else:    
            return TrafficLight.UNKNOWN
        