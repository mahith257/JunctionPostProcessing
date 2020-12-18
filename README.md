# JunctionPostProcessing


Steps:

1. Take binary mask or instance mask images

2. Slice the images into small pixels blocks of size 7*7 or such

3. Apply HoughLinesP method in each and every block to detect the lines

4. If any two lines in the same slice are within 10% slope then create a median of the two lines(removing the original two lines) --- R1

5. Take the lines detected till now including the median lines and get the points on each and every at 10px or such interval

6. Associate the points with respect to their corresponding slice

7. Apply polfit method in each and every slice

Steps 5, 6, 7 --- R2

8. Assemble the slices back

9. Draw the fitted line using polyfit method in the assembled image

10. Write the assembled images after step 9 into the folder

11. Write the visualized images too into the folder


Pipelines which can be implemented with the above steps:

1. Binary mask - slicing - HoughLinesP - R1 - Assembled back - R2

2. Binary mask - erosion - slicing - HoughLinesP - R1 - Assembled back - R2

3. Binary mask - skeleton - slicing - HoughLinesP - R1 - Assembled back - R2

4. Instance mask - slicing - HoughLinesP - R1 - Assembled back - R2

5. Instance mask - erosion - slicing - HoughLinesP - R1 - Assembled back - R2

6. Instance mask - skelton - slicing - HoughLinesP - R1 - Assembled back - R2