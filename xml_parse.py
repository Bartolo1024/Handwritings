import xml.etree.ElementTree as ET

ESC_KEY = 27

def coordinates_generator(XML_name, window_size, image_size, skip_elements_smaller_than = 50):
    """
    funtion yields box coordinates [x1, y1, x2, y2]

    @param XML_name - name of XML file
    @param widow_size - tuple of (x, y) - size of image boxes you want to get
    @param image_size - np.shape(your_image) ;)
    @skip_elements_smaller_than - used to skip commas and such
    """
    tree = ET.parse(XML_name)
    root = tree.getroot()

    image_size_x = image_size[0]
    image_size_y = image_size[1]

    x_bound = window_size[0]//2 + 1
    y_bound = window_size[1]//2 + 1

    for word in root.iter('word'):
        for c in word.iter('cmp'):
            
            if(int(c.attrib['height']) < skip_elements_smaller_than and int(c.attrib['width'])<skip_elements_smaller_than):
                # maly obrazek, pewnie nie warto
                continue
            
            # srodek boxa
            center_x = int(c.attrib['x']) + int(c.attrib['width'])//2
            center_y = int(c.attrib['y']) + int(c.attrib['height'])//2
            

            # miejsce na sprawdzenie czy boxy nie będą wykraczać poza granice i przesunięcie
            center_x = max(center_x, x_bound)
            center_y = max(center_y, y_bound)
            center_x = min(center_x, image_size_x - x_bound)
            center_y = min(center_y, image_size_y - y_bound)

            yield [center_y - y_bound, center_x - x_bound, center_y + y_bound, center_x + x_bound]


if(__name__ == '__main__'):
    import cv2
    # wczytanie obrazka    
    img = cv2.imread('data/formsA-D/a01-007.png')

    # lecimy po wszystkich boxach z xmla
    for box in coordinates_generator('data/xml/a01-007.xml', [500, 150], img.shape):
        cv2.imshow(str(box), img[box[0]: box[2], box[1]: box[3]])
        key = cv2.waitKey(0)
        if key is ESC_KEY:
            break

















