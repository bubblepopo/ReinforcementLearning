import imageio
import cv2

class VideoRecorder(object):
    def __init__(self):
        self.init()

    def init(self):
        self.frames = []
        
    def record(self, env):
        frame = env.render(mode='rgb_array')
        frame = cv2.resize(frame, (256,256))
        # cv2.imshow('..', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.frames.append(frame)

    def save(self, filepath):
        imageio.mimsave(filepath, self.frames, fps=30)
