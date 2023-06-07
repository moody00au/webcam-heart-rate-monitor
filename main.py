import cv2
import numpy as np
import sys
import mediapipe as mp

# initialize BPM measurements per frame
bpmmeasurements = np.empty((0, 2))

# Add Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Helper Methods
def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid

def reconstructFrame(pyramid, index, levels):
    filteredFrame = pyramid[index]
    for level in range(levels):
        filteredFrame = cv2.pyrUp(filteredFrame)
    filteredFrame = filteredFrame[:videoHeight, :videoWidth]
    return filteredFrame

# Webcam Parameters
webcam = None
if len(sys.argv) == 2:
    webcam = cv2.VideoCapture(sys.argv[1])
else:
    webcam = cv2.VideoCapture(0)
    
realWidth = 640
realHeight = 480
videoWidth = 640
videoHeight = 480
videoChannels = 3
videoFrameRate = 30
webcam.set(3, realWidth)
webcam.set(4, realHeight)

# Output Videos
if len(sys.argv) != 2:
    originalVideoFilename = "original.avi"
    originalVideoWriter = cv2.VideoWriter()
    originalVideoWriter.open(originalVideoFilename, cv2.VideoWriter_fourcc('j', 'p', 'e', 'g'), videoFrameRate, (realWidth, realHeight), True)

outputVideoFilename = "BPM.avi"
outputVideoWriter = cv2.VideoWriter()
outputVideoWriter.open(outputVideoFilename, cv2.VideoWriter_fourcc('j', 'p', 'e', 'g'), videoFrameRate, (realWidth, realHeight), True)

# Color Magnification Parameters
levels = 3
alpha = 170
minFrequency = 1.0
maxFrequency = 2.0
bufferSize = 150
bufferIndex = 0

# Output Display Parameters
font = cv2.FONT_HERSHEY_SIMPLEX
loadingTextLocation = (20, 30)
bpmTextLocation = (videoWidth//2 + 5, 30)
fontScale = 0.5  # Modify the font scale to make the text half the size
fontColor = (0, 255, 0)
lineType = 2
boxColor = (0, 255, 0)
boxWeight = 3

# Initialize Gaussian Pyramid
firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
firstGauss = buildGauss(firstFrame, levels+1)[levels]
videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
fourierTransformAvg = np.zeros((bufferSize))

# Bandpass Filter for Specified Frequencies
frequencies = (1.0*videoFrameRate) * np.arange(bufferSize) / (1.0*bufferSize)
mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

# Heart Rate Calculation Variables
bpmCalculationFrequency = 15
bpmBufferIndex = 0
bpmBufferSize = 10
bpmBuffer = np.zeros((bpmBufferSize))

i = 0 # Initialize i for BPM calculation

while (True):
    ret, frame = webcam.read()
    if ret == False:
        break

    if len(sys.argv) != 2:
        originalFrame = frame.copy()
        originalVideoWriter.write(originalFrame)

    # Use MediaPipe Face Detection
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_detection.process(image)

    # Check if faces are detected
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Ignore if bounding box is too close to the edge
            if bbox[0] < 10 or bbox[1] < 10 or bbox[0] + bbox[2] > iw - 10 or bbox[1] + bbox[3] > ih - 10:
                continue
                
            detectionFrame = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2], :]
            detectionFrame = cv2.resize(detectionFrame, (videoWidth, videoHeight)) # resize detectionFrame

            # Construct Gaussian Pyramid
            videoGauss[bufferIndex] = buildGauss(detectionFrame, levels+1)[levels]
            fourierTransform = np.fft.fft(videoGauss, axis=0)

            # Bandpass Filter
            fourierTransform[mask == False] = 0

            # Grab a Pulse
            if bufferIndex % bpmCalculationFrequency == 0:
                i = i + 1
                for buf in range(bufferSize):
                    fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
                hz = frequencies[np.argmax(fourierTransformAvg)]
                bpm = 60.0 * hz
                bpmBuffer[bpmBufferIndex] = bpm
                bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize
            

            # Amplify
            filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
            filtered = filtered * alpha

            # Reconstruct Resulting Frame
            filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
            outputFrame = detectionFrame + filteredFrame
            outputFrame = cv2.convertScaleAbs(outputFrame)

            bufferIndex = (bufferIndex + 1) % bufferSize

            frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2], :] = cv2.resize(outputFrame, (bbox[2], bbox[3])) # resize outputFrame to fit original bbox size
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), boxColor, boxWeight)

            # Display BPM
            if i > bpmBufferSize:
                # Update BPM Array with the latest BPM and frame number
                bpmmeasurements = np.append(bpmmeasurements, [[i, bpmBuffer.mean()]], axis=0)

                # Calculate the average BPM and the lowest and highest BPM
                averageBPM = bpmmeasurements[:, 1].mean()
                lowestBPM = bpmmeasurements[:, 1].min()
                highestBPM = bpmmeasurements[:, 1].max()

                # Display the average BPM, lowest BPM, and highest BPM
                cv2.putText(frame, "BPM: %d" % bpmBuffer.mean(), (20, 60), font, 0.5, fontColor, lineType)
                cv2.putText(frame, "Average BPM: %d" % averageBPM, (20, 90), font, 0.5, fontColor, lineType)
                cv2.putText(frame, "Lowest BPM: %d" % lowestBPM, (20, 120), font, 0.5, fontColor, lineType)
                cv2.putText(frame, "Highest BPM: %d" % highestBPM, (20, 150), font, 0.5, fontColor, lineType)
        
                
            
            else:
                cv2.putText(frame, "Calculating BPM...", (20, 30), font, 0.5, fontColor, lineType)

            outputVideoWriter.write(frame)

            if len(sys.argv) != 2:
                cv2.imshow("Webcam Heart Rate Monitor", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

webcam.release()
cv2.destroyAllWindows()
outputVideoWriter.release()
if len(sys.argv) != 2:
    originalVideoWriter.release()
