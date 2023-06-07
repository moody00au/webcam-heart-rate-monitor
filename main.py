    # Once BPM is calculated for 2 seconds, calculate the average, lowest, and highest BPM
    if i >= 30:
        # Calculate the average BPM and the lowest and highest BPM
        averageBPM = bpmmeasurements[:, 1].mean()
        lowestBPM = bpmmeasurements[:, 1].min()
        highestBPM = bpmmeasurements[:, 1].max()

    # Display the current BPM, average BPM, lowest BPM, and highest BPM
    cv2.putText(frame, "Current BPM: %d" % bpmBuffer.mean(), (20, 60), font, 0.5, fontColor, lineType)
    cv2.putText(frame, "Average BPM: %d" % averageBPM, (20, 90), font, 0.5, fontColor, lineType)
    cv2.putText(frame, "Lowest BPM: %d" % lowestBPM, (20, 120), font, 0.5, fontColor, lineType)
    cv2.putText(frame, "Highest BPM: %d" % highestBPM, (20, 150), font, 0.5, fontColor, lineType)
