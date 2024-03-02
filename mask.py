import cv2
import numpy as np
import click
import os

# USAGE:
# PRESS Q TO FINISH
# PRESS LEFT CLICK TO DRAW MASK

def draw_mask(image):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask_color = np.zeros(image.shape, dtype=np.uint8)
    mask_color[:, :] = [255, 0, 200]
    drawing = False
    ix, iy = -1, -1

    def draw(event, x, y, flags, param):
        nonlocal drawing, ix, iy
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.circle(mask, (x, y), 10, (1), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.circle(mask, (x, y), 10, (1), -1)

    cv2.namedWindow("Draw Mask")
    cv2.setMouseCallback("Draw Mask", draw)

    while True:
        mask_3d = mask[:,:, np.newaxis]
        mask_3d = np.repeat(mask_3d, 3, axis=2)

        cv2.imshow("Draw Mask", image*(1-mask_3d) + mask_3d*mask_color)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    return mask

@click.command()
@click.pass_context
@click.option('--target', '-t', 'target_fname', type=click.Path(exists=True, dir_okay=False), help='input image to mask', required=True, metavar='FILE')
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'experiences', 'out', 'mask'), show_default=True, metavar='DIR')
def main(ctx: click.Context, target_fname: str, outdir: str):
    file_name = target_fname.split("\\")[-1]
    image = cv2.imread(target_fname)  # Replace "input_image.jpg" with your image file path
    mask = draw_mask(image)
    cv2.imwrite(outdir + file_name, np.round(mask*255))  # Save the mask as an image file
    np.save(outdir + file_name[0:-4], mask)

if __name__ == "__main__":
    main()
