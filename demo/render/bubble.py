import cv2, textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

class ResponseBubble:
    def __init__(
        self,
        font_path: str = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        font_size: int = 50,
        meta_font_path: str = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        meta_font_size: int = 25
    ):
        self.font = ImageFont.truetype(font_path, font_size)
        self.meta_font = ImageFont.truetype(meta_font_path, meta_font_size)

    def draw_bubble(
        self,
        base_img: Image.Image,
        position: tuple,
        text: str,
        metadata: str,
        padding: int = 15,
        line_spacing: int = 8,
        radius: int = 20,
        bg_color: tuple = (255, 255, 255, 200),  # translucent white
        text_color: tuple = (0, 0, 0, 255),
        meta_color: tuple = (50, 50, 50, 200),
        blur_radius: int = 8
    ):
        # Ensure image is RGBA
        base = base_img.convert('RGBA')
        overlay = Image.new('RGBA', base.size)
        draw = ImageDraw.Draw(overlay, 'RGBA')
        x, y = position

        # Wrap text and calc sizes
        wrapped = textwrap.wrap(text, width=50)
        meta_w, meta_h = self.meta_font.getbbox(metadata)[2:]
        line_sizes = [self.font.getbbox(line)[2:] for line in wrapped]
        max_w = max([meta_w] + [w for w, _ in line_sizes])
        total_h = meta_h + sum(h for _, h in line_sizes) + line_spacing * len(wrapped)

        bubble_w = max_w + 2 * padding
        bubble_h = total_h + 2 * padding
        box = (x, y, x + bubble_w, y + bubble_h)

        # Blur background behind bubble
        region = base.crop(box).filter(ImageFilter.GaussianBlur(blur_radius))
        overlay.paste(region, box)

        # Draw rounded rect
        draw.rounded_rectangle(box, radius=radius, fill=bg_color)

        # Draw text & metadata
        tx, ty = x + padding, y + padding
        draw.text((tx, ty), metadata, font=self.meta_font, fill=meta_color)
        ty += meta_h + line_spacing
        for line in wrapped:
            draw.text((tx, ty), line, font=self.font, fill=text_color)
            ty += self.font.getbbox(line)[3] + line_spacing

        # Composite onto base
        composed = Image.alpha_composite(base, overlay)
        return composed.convert('RGB')

class QueryBubble(ResponseBubble):
    def draw_bubble(
        self,
        base_img: Image.Image,
        position: tuple,
        text: str,
        metadata: str = '',
        padding: int = 15,
        line_spacing: int = 8,
        radius: int = 20,
        bg_color: tuple = (173, 216, 230, 150),  # translucent light blue
        text_color: tuple = (0, 0, 0, 255),
        meta_color: tuple = (50, 50, 50, 200),
        blur_radius: int = 8
    ):
        # Ensure RGBA
        base = base_img.convert('RGBA')
        overlay = Image.new('RGBA', base.size)
        draw = ImageDraw.Draw(overlay, 'RGBA')
        x, y = position

        # Wrap text
        wrapped = textwrap.wrap(text, width=50)
        meta_w, meta_h = (self.meta_font.getbbox(metadata)[2:] if metadata else (0, 0))
        sizes = [self.font.getbbox(line)[2:] for line in wrapped]
        max_w = max([meta_w] + [w for w, _ in sizes])
        total_h = (meta_h if metadata else 0) + sum(h for _, h in sizes) + line_spacing * len(wrapped)

        bubble_w = max_w + 2 * padding
        bubble_h = total_h + 2 * padding

        # Tail adjustment
        tail_h = 12
        box = (x, y, x + bubble_w, y + bubble_h)
        tail = [(x, y + 20), (x - tail_h, y + 30), (x, y + 40)]

        # Blur region including tail
        min_x = min(box[0], tail[1][0])
        min_y = min(box[1], tail[0][1])
        max_x = box[2]
        max_y = max(box[3], tail[2][1])
        blur_box = (min_x, min_y, max_x, max_y)
        region = base.crop(blur_box).filter(ImageFilter.GaussianBlur(blur_radius))
        overlay.paste(region, blur_box)

        # Draw tail and bubble
        draw.polygon(tail, fill=bg_color)
        draw.rounded_rectangle(box, radius=radius, fill=bg_color)

        # Draw text
        tx, ty = x + padding, y + padding
        if metadata:
            draw.text((tx, ty), metadata, font=self.meta_font, fill=meta_color)
            ty += meta_h + line_spacing
        for line in wrapped:
            draw.text((tx, ty), line, font=self.font, fill=text_color)
            ty += self.font.getbbox(line)[3] + line_spacing

        # Composite
        composed = Image.alpha_composite(base, overlay)
        return composed.convert('RGB')

if __name__ == '__main__':
    video_path = 'demo/sources/writing_mute.mp4'
    # read the first image
    cap = cv2.VideoCapture(video_path)
    ret, img = cap.read()
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Response
    resp = ResponseBubble()
    pil = resp.draw_bubble(pil, (50, 200), "This is the chatbot's response.", "Time: 0.2s | Model: LiveCC-7B-Instruct")

    # Query
    qry = QueryBubble()
    pil = qry.draw_bubble(pil, (50, 50), "What is the weather today?", "User")

    out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite('test.png', out)