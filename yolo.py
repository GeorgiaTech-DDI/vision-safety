import cv2
from ultralytics import YOLO

def run_webcam_detection():
    model = YOLO('runs/detect/train/weights/best.pt')
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        
        if success:
            # Run inference on the frame
            results = model(frame)
            
            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            
            # Display the annotated frame
            cv2.imshow("PPE Detection", annotated_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_detection()
