# Assignment 02: ANPR Dataset Creation from Video Footage

## tools

- python
- opencv
- yolo11
- openvino: for gpu optimization

## challenges

the first script to automate the extraction of vide frames by detecting if there was a vehicle was taking more that five hours to process more than 50k frames so I had to optimize the script to take advantage of the intel gpu in our laptop but the integration was not straightforward which ended up taking a lot of time to get it to work😅

## expericence

because the video was more than one hour, it meant that it was not a good idea to use manual method so I had to automate it using yolo11 the latest model in the yolo model series which is faster and more accurate, also I had to add functions like detect if  the vehicle is close enough and if the frame is not blurry. this allowed to remain with about 3K frames in about 50K frames which I then went through and sorted them accordingly.

this helped me learn how to use tools to simply the process of dataset creation and how to work with models like YOLO11 and other tools like openvino.

**Benjamin ishimwe mugisha**
