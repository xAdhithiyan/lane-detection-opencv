import cv2
import matplotlib.pyplot as plt
import numpy as np

def canny_(image):
    #edge detection->identifying sharp changes in intensity in adjacent pixels
    #strong gradient(0->255)(step change),small gradient(0->15)(shallow change)
    #gradient image is obtained from grayscale rather then original -> processing is easier 
    img_gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    #reducing noise by smoothing to increase accuracy
    img_blur=cv2.GaussianBlur(img_gray,(5,5),0)
    #canny->measures changes in intensity in all direction,x and y
    #it traces the edge with large change in intensity(large gradient) in an ontline of white pixels
    canny = cv2.Canny(img_blur , 50 ,150)
    return canny

#tracing(masking) a triangle(lane) on a black image
def region_of_intrest(image):
    height = image.shape[0]
    #apprrox coordiantes of the lanes 
    polygon=np.array(
        [[(200,height),(1100,height),(550,250)]]
        )
    mask=np.zeros_like(image) #creating a black image from the original image
    cv2.fillPoly(mask,polygon,color=(255,255,255))
    
    #binary representation of white image is 11111111 and black image is 0000
    #apply bitwise and for the masked image and canny image to get region of intrest
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image


def display_lines(image,lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image


#returns coordintaes by taking in (slope and y intercept) as parameters
def make_coordinates(image,line_parameters):
    slope,intercept=line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5)) #we want this detection to be applied for only 3/5 of the image from the bottom 
    x1 =int((y1 - intercept)/slope) #from eqn of line y = mx + c
    x2= int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(lane_img,lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        #np.polyfit returns the slope and y intercept of the line
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0: #left lane will have negative slope(because y axis is downwards in pyplotlib)
            left_fit.append((slope,intercept))
        else: #right lane will have positive slope 
            right_fit.append((slope,intercept))
    left_fit_average=np.average(left_fit,axis=0)#axis=0->averaging by columns
    right_fit_average=np.average(right_fit,axis=0)
    left_line=make_coordinates(lane_img,left_fit_average)
    right_line=make_coordinates(lane_img,right_fit_average)
    return np.array([left_line,right_line])

img =cv2.imread(r"lane-detection/test_image.jpg")


''''lane_img=np.copy(img) 
canny_image=canny_(img)
masked_image=region_of_intrest(canny_image)

#hough space->a line(in cartessian plane) is plotted as a point(in hough space) with coordinates (y intercept,slope)
#hough space->a point(in cartessian plane) can be represented by a line(in hough space)
#hough space->multiple points(in cartessian plane) are plotted by different lines(in hough space).the intersection point(in hough space) gives the the line that passes through the multiple points(in cartessian plane )
#since slope of vertical lines is infinite it cannt be  ploted on the hough space. therefore we use polar coordinates(eqn of line->xcos(theta)+ysin(theta))
#for polar coordinates the first 3 points are same expect the points are represented as curves in hough space

#cv2.HoughLinesP(image, rho, theta, threshold[,lines[,minLineLength[,maxLineGap]]]) \
#HoughLinesP contain all the possible lines with threshold above 100
lines=cv2.HoughLinesP(masked_image,2,np.pi/180 ,100,np.array([]), minLineLength=40,maxLineGap=5) 
averaged_lines=average_slope_intercept(lane_img,lines)
#we are passing averaged_lines
line_image=display_lines(lane_img,averaged_lines)
#adding both the images with line_image 20% more weight 
final_image=cv2.addWeighted(lane_img,0.8,line_image,1 ,1)
cv2.imshow("img",final_image)
cv2.waitKey(0)
'''


#adding vid
cap = cv2.VideoCapture("lane-detection/test2.mp4")
while(cap.isOpened()):
    reg_,frame = cap.read()
    canny_image=canny_(img)
    masked_image=region_of_intrest(canny_image)

    #hough space->a line(in cartessian plane) is plotted as a point(in hough space) with coordinates (y intercept,slope)
    #hough space->a point(in cartessian plane) can be represented by a line(in hough space)
    #hough space->multiple points(in cartessian plane) are plotted by different lines(in hough space).the intersection point(in hough space) gives the the line that passes through the multiple points(in cartessian plane )
    #since slope of vertical lines is infinite it cannt be  ploted on the hough space. therefore we use polar coordinates(eqn of line->xcos(theta)+ysin(theta))
    #for polar coordinates the first 3 points are same expect the points are represented as curves in hough space

    #cv2.HoughLinesP(image, rho, theta, threshold[,lines[,minLineLength[,maxLineGap]]]) \
    #HoughLinesP contain all the possible lines with threshold above 100
    lines=cv2.HoughLinesP(masked_image,2,np.pi/180 ,100,np.array([]), minLineLength=40,maxLineGap=5) 
    averaged_lines=average_slope_intercept(frame,lines)
    #we are passing averaged_lines
    line_image=display_lines(frame,averaged_lines)
    #adding both the images with line_image 20% more weight 
    final_image=cv2.addWeighted(frame,0.8,line_image,1 ,1)
    cv2.imshow("img",final_image)
    if cv2.waitKey(1) == ord('q'):
        break;
cap.release()
cv2.destroyAllWindows() 