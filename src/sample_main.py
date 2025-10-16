import cv2

class Button:
    def __init__(self, pos, width, height, value):
        self.pos = pos
        self.width = width
        self.height = height
        self.value = value

    def draw(self, img):
        cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height),
                      (225, 225, 225), cv2.FILLED)
        cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height),
                      (50, 50, 50), 3)

        text_x = self.pos[0] + self.width // 2 - 15
        text_y = self.pos[1] + self.height // 2 + 15

        cv2.putText(img, self.value, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN,
                    3, (50, 50, 50), 3)


# MUST be configured for each system
cap = cv2.VideoCapture(2) # here would be 0 or 1 for most systems, but mine is 2
# it is the webcam address, in my case in /dev/video2

#input image size, based on your webcam
WIDTH = 1920
HEIGHT = 1080
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# basic list of buttons, may be changed (changes must be justified)
buttonListValues = [['C', '<'],
                    ['7', '8', '9', '/'],  # ÷ cant be rendered, so well have to do with / :/
                    ['4', '5', '6', '*'],
                    ['1', '2', '3', '-'],
                    ['0', '.', '=', '+']]

buttonlist = []
for y in range(5):
    for x in range(4):
        if y == 0 and x >= 2:  # first row only has 2 buttons (C and <), so kinda special case
            break

        xpos = int(WIDTH - 500 + x * 100)
        ypos = int(HEIGHT * 0.15 + y * 100)

        # first row special case, second button needs to be shifted
        if y == 0 and x == 1: #not the most elegant
            xpos += 100

        if y == 0:
            width = 200
            buttonlist.append(Button((xpos, ypos), width, 100, buttonListValues[y][x]))
        else:
            buttonlist.append(Button((xpos, ypos), 100, 100, buttonListValues[y][x]))

operation = ""

while True:
    success, img = cap.read()

    operation_x = int(WIDTH - 500)
    operation_y = int(HEIGHT * 0.05)

    cv2.rectangle(img, (operation_x, operation_y), (operation_x + 400, operation_y + 120),
                  (225, 225, 225), cv2.FILLED)
    cv2.rectangle(img, (operation_x, operation_y), (operation_x + 400, operation_y + 120),
                  (50, 50, 50), 3)

    for button in buttonlist:
        button.draw(img)

    # when something triggers, the calculator should actually calculate
    # the something should be programmed by you, of course
    something = False
    received_val = ""
    if something:
        if received_val == "=":
            try:
                operation = str(eval(operation))
            except:
                operation = "Error"
        elif received_val == "C":  # Reset
            operation = ""
        elif received_val == "<":  # remove char
            operation = operation[:-1]
        else:
            operation += received_val
        delayCounter = 1

    cv2.putText(img, operation, (operation_x + 10, operation_y + 75),
                cv2.FONT_HERSHEY_PLAIN, 3, (50, 50, 50), 3)

    cv2.imshow('Calculator', img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

