import cv2

for i in range(1, 157):
    gamePlay_img = cv2.imread('test.png', cv2.IMREAD_REDUCED_COLOR_2)
    box_img = cv2.imread(f"pics/a ({str(i)}).png", cv2.IMREAD_REDUCED_COLOR_2)

    result = cv2.matchTemplate(gamePlay_img, box_img, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    threshold = 0.7

    if max_val >= threshold:
        print("found box")

        box_width = box_img.shape[1]
        box_height = box_img.shape[0]

        top_left = max_loc
        bottom_right = (top_left[0] + box_width, top_left[1] + box_height)

        cv2.rectangle(gamePlay_img, top_left, bottom_right,
                      color=(0, 255, 0), thickness=2, lineType=cv2.LINE_4)
        cv2.imshow('Result', gamePlay_img)
        cv2.waitKey()





