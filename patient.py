
class Patient:
    def __init__(self):
        self.images = []
        self.contours = []

    def add_data(self, image, contour):
        self.images.append(image)
        self.contours.append(contour)

    def is_empty(self):
        return len(self.images) == 0 and len(self.contours) == 0
