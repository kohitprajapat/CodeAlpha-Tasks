from ultralytics import YOLO
import cv2

# load the YOLO model
model = YOLO('best.pt')  # best.pt is the model trained in Google Colab

# function for live webcam detection
def live_detection():
    # set detection threshold
    threshold = 0.5

    # open the webcam (0 is the default camera, change to 1, 2, etc., for other cameras)
    cap = cv2.VideoCapture(0)

    # check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    print("\nStarting live detection...... Press 'q' to exit.")
    
    # for live camera feed
    while True:
        # capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Unable to read the camera feed.")
            break

        # run YOLO detection on the frame
        results = model(frame)[0]

        # process detection results
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            # check if the score exceeds the detection threshold
            if score > threshold:  
                # draw bounding box around the detected object
                top_left = (int(x1), int(y1))  # top-left corner of the bounding box
                bottom_right = (int(x2), int(y2))  # bottom-right corner of the bounding box
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)  # draw the rectangle

                # create a label for the detected object
                class_name = results.names[int(class_id)].upper()  # get the class name and convert to uppercase
                confidence_score = f"({score:.2f})"  # format the confidence score
                label = f"{class_name} {confidence_score}"  # combine class name and confidence score

                # add the label to the frame above the bounding box
                label_position = (int(x1), int(y1) - 10)  # position of the label
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # draw the text


        # display the resulting frame
        cv2.imshow('YOLO Live Detection', frame)

        # press 'q' to exit the live feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # release resources
    cap.release()
    cv2.destroyAllWindows()

# function for static image prediction
def static_image_prediction(image_path):
    print(f"\nProcessing image: {image_path}")
    model.predict(source=image_path, imgsz=640, conf=0.5, save=True)
    print("Static image prediction completed.")

# main function to choose mode
def main():
    print("================================================================")
    print("          Welcome to YOLO Object Detection and Tracking                ")
    print("================================================================")
    mode = input("Write 'live' for live detection or 'static' for static image prediction: ").strip().lower()
    
    if mode == 'live':
        live_detection()
    elif mode == 'static':
        image_path = input("Enter the path to the image file: ")
        static_image_prediction(image_path)
    else:
        print("Invalid input. Please enter 'live' or 'static'.")

if __name__ == "__main__":
    main()
