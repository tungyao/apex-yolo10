from ctypes import windll
MOUSE_EVEN_TF_LEFT_DOWN = 0x2
MOUSE_EVEN_TF_LEFT_UP = 0x4
MOUSE_EVEN_TF_MIDDLE_DOWN = 0x20
MOUSE_EVEN_TF_MIDDLE_UP = 0x40
MOUSE_EVEN_TF_RIGHT_DOWN = 0x8
MOUSE_EVEN_TF_RIGHT_UP = 0x10
MOUSE_EVEN_TF_MOVE = 0x1

user32 = windll.user32
def mouse_move(x,y):
    user32.mouse_event(MOUSE_EVEN_TF_MOVE, int(x), int(y))

if __name__== "__main__":
    mouse_move(100,100)