import logging
logger = logging.getLogger(__name__)

class BoundingBox:
    def __init__(self, *args, **kwargs):
        '''
        type1: x1, y1, x2, y2
        type2: (x1, y1), (x2, y2)
        type3: x1, y1, width=w, height=h
        type4: (x1, y1), width=w, height=h
        '''
        if len(args) == 4:
            self.x1, self.y1, self.x2, self.y2 = args
        elif len(args) == 2 and isinstance(args[0], tuple) and isinstance(args[1], tuple):
            self.x1, self.y1 = args[0]
            self.x2, self.y2 = args[1]
        elif len(args) == 2 and 'width' in kwargs and 'height' in kwargs:
            self.x1, self.y1 = args[0]
            self.x2 = self.x1 + kwargs['width']
            self.y2 = self.y1 + kwargs['height']
        elif len(args) == 1 and isinstance(args[0], tuple) and 'width' in kwargs and 'height' in kwargs:
            self.x1, self.y1 = args[0]
            self.x2 = self.x1 + kwargs['width']
            self.y2 = self.y1 + kwargs['height']
        else:
            raise ValueError("Invalid arguments for BoundingBox")
        if self.x1 > self.x2 or self.y1 > self.y2:
            raise ValueError("Invalid coordinates for BoundingBox")

    def __str__(self):
        return f"({self.x1}, {self.y1}), ({self.x2}, {self.y2})"

    def __repr__(self):
        return f"BoundingBox({self.x1}, {self.y1}, {self.x2}, {self.y2})"

    def __eq__(self, other):
        return self.x1 == other.x1 and self.y1 == other.y1 and self.x2 == other.x2 and self.y2 == other.y2

    def __hash__(self):
        return hash((self.x1, self.y1, self.x2, self.y2))

    def width(self):
        return abs(self.x2 - self.x1)

    def height(self):
        return abs(self.y2 - self.y1)

    def area(self):
        return self.width() * self.height()

    def center(self):
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def intersect(self, other):
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        if x1 >= x2 or y1 >= y2:
            return None
        return BoundingBox(x1, y1, x2, y2)
    
    def topleft(self):
        return (self.x1, self.y1)
    
    def bottomright(self):
        return (self.x2, self.y2)
    
    def topright(self):
        return (self.x2, self.y1)
    
    def bottomleft(self):
        return (self.x1, self.y2)
    
    def top(self):
        return self.y1
    
    def bottom(self):
        return self.y2
    
    def left(self):
        return self.x1
    
    def right(self):
        return self.x2
