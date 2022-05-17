import cv2
import math

from PythonUser.detectme.models import line_list

def onMouse(event, x, y, flags, param):
    selected = line_select(x, y, param)

    if flags == cv2.EVENT_FLAG_LBUTTON:
        if event == cv2.EVENT_LBUTTONDOWN:
            i = len(param)
            on_mouse = False
            start = (x, y)
            end = (x, y)
            to_left = []
            to_right = []
            li = [i, on_mouse, [start, end], to_left, to_right]
            param.append(li)
        else:
            if len(param) > 0:
                li = param[len(param)-1]
                li[2][1] = (x,y)
    else:
        if len(param) > 0:
            li = param[len(param)-1]
            if event == cv2.EVENT_LBUTTONUP:
                if li[2][0] == li[2][1]:
                    del param[len(param)-1]

    if event == cv2.EVENT_RBUTTONDOWN:
        temp = []
        for i in range(len(param)):
            if not param[i][1]:
                temp.append(param[i])
                temp[len(temp)-1][0] = len(temp)-1
        while len(param) > 0:
            param.pop()
        for i in range(len(temp)):
            param.append(temp[i])

def line_select(x, y, lines):
    selected = False
    for i, on_mouse, line, to_left, to_right in lines:
        sx = line[0][0]
        sy = line[0][1]
        ex = line[1][0]
        ey = line[1][1]
        if (x < sx and x < ex) or (x > sx and x > ex) or (y < sy and y < ey) or (y > sy and y > ey):
            lines[i][1] = False
        else:
            if line[0] != line[1]:
                distance = abs((ex - sx) * (sy - y) - (sx - x) * (ey - sy)) / ((ex-sx)**2 + (ey-sy)**2)**0.5
                if distance < 5:
                    lines[i][1] = True
                    selected = True
                else:
                    lines[i][1] = False
    return selected

def draw_lines(img, lines):
    #print(lines)
    for i, on_mouse, line, to_left, to_right in lines:
        if on_mouse:
            cv2.line(img, line[0], line[1], (0, 255, 0), 3)
        else:
            cv2.line(img, line[0], line[1], (255, 255, 255), 2)

def line_count(img, lines):
    for i, on_mouse, line, to_left, to_right in lines:
        if line[1][1] - line[0][1] == 0:
            radian = math.radians(90)
        elif line[1][0] - line[0][0] == 0:
            radian = math.radians(0)
        else:
            degree = (line[1][1] - line[0][1]) / (line[1][0] - line[0][0])
            degree = -1 / degree
            radian = math.atan(degree)
        
        center = [(line[0][0] + line[1][0]) / 2, (line[0][1] + line[1][1]) / 2]

        #print(radian)

        if line[0][1] > line[1][1]:
            direction = 1
        elif line[0][1] == line[1][1]:
            if line[0][0] > line[1][0]:
                direction = -1
            else:
                direction = 1
        else:
            direction = -1
            
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        text1 = str(len(to_left))
        size, baseline = cv2.getTextSize(text1, font, 1, 1)
        letter1_x = int(center[0] + 30 * math.cos(radian) * direction - size[0]/2)
        letter1_y = int(center[1] + 30 * math.sin(radian) * direction + size[1]/2)
        
        text2 = str(len(to_right))
        size, baseline = cv2.getTextSize(text2, font, 1, 1)
        letter2_x = int(center[0] - 30 * math.cos(radian) * direction - size[0]/2)
        letter2_y = int(center[1] - 30 * math.sin(radian) * direction + size[1]/2)
        
        cv2.putText(img, text1, (letter1_x, letter1_y), font, 1, (255, 255, 255), 2)
        cv2.putText(img, text2, (letter2_x, letter2_y), font, 1, (255, 255, 255), 2)
    
def line_manage(im0, lines):

    win_name = "Line_Manage"
    cv2.namedWindow(win_name)

    cv2.setMouseCallback(win_name, onMouse, param=lines)
    key = 0
    while key != 113:
        img = im0.copy()
        draw_lines(img, lines)
        cv2.imshow(win_name, img)
        key = cv2.waitKey(10)

    for i in range(len(lines)):
        lines[i][1] = False
    cv2.destroyAllWindows()
'''
line 저장 (중복 제거)
2022.05.16 박병제
'''           
def line_save(lines):
    for i, on_mouse,line,to_left,to_right in lines:
        if line_list.objects.filter(line_number = int(i)).exists():
            line_delete = line_list.objects.get(line_number = int(i))
            line_delete.delete()
        line_list.objects.create(line_number= int(i),line_one_x=line[0][0],line_one_y=line[0][1],line_two_x=line[1][0],line_two_y=line[1][1])

def check_cross(obj_num, tails, lines):
    if len(tails) > 0:
        for i, on_mouse, line, to_left, to_right in lines:
            for t in range(len(tails)-1):
                start = tails[t]
                end = tails[t+1]
                if line_cross(line[0][0], line[0][1], line[1][0], line[1][1], start[0], start[1], end[0], end[1]) == 1:
                    lx1 = line[0][0]
                    ly1 = line[0][1]
                    lx2 = line[1][0]
                    ly2 = line[1][1]
                    in_out = ccw(lx1, ly1, lx2, ly2, start[0], start[1])

                    #print(in_out)
                    
                    if in_out > 0 and obj_num not in to_left:
                        #if obj_num in to_right:
                        #    del to_right[to_right.index(obj_num)]
                        to_left.append(obj_num)
                        
                    elif in_out < 0 and obj_num not in to_right:
                        #if obj_num in to_left:
                        #    del to_left[to_left.index(obj_num)]
                        to_right.append(obj_num)

def line_cross(x1, y1, x2, y2, x3, y3, x4, y4):
    mx1, my1, mx2, my2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    mx3, my3, mx4, my4 = min(x3, x4), min(y3, y4), max(x3, x4), max(y3, y4)
    
    ccw123 = ccw(x1, y1, x2, y2, x3, y3)
    ccw124 = ccw(x1, y1, x2, y2, x4, y4)
    ccw341 = ccw(x3, y3, x4, y4, x1, y1)
    ccw342 = ccw(x3, y3, x4, y4, x2, y2)

    # 평행
    if ccw123*ccw124 == 0 and ccw341*ccw342 == 0:
        if mx1 <= mx4 and mx3 <= mx2 and my1 <= my4 and my3 <= my2:
            return 1
    # 교차
    else:
        if ccw123*ccw124 <= 0 and ccw341*ccw342 <= 0:
            return 1

    return 0

def ccw(x1, y1, x2, y2, x3, y3):
    return (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)

"""
def main():
    video_path = 'OTtest.mp4'
    cap = cv2.VideoCapture(video_path)
    
    lines = []

    ret, img = cap.read()
    line_manage(img, lines)

    tails = []
    for i in range(30):
        tails.append([10*i, 10*i])
    check_cross(0, tails, lines)
    
    draw_lines(img, lines)
    line_count(img, lines)
    
    win_name = "Line_Manage"
    cv2.namedWindow(win_name)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)
    print(lines)

main()
"""
