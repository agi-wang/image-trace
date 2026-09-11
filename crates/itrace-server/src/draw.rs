//! Match visualization: side-by-side canvas with keypoint circles and
//! match lines — replacement for cv2.drawMatches.

use itrace_core::descriptors::{DescriptorSet, Match};
use itrace_core::RgbImage;

fn put_px(img: &mut RgbImage, x: i32, y: i32, c: (u8, u8, u8)) {
    if x >= 0 && y >= 0 && x < img.width as i32 && y < img.height as i32 {
        let i = ((y as u32 * img.width + x as u32) * 3) as usize;
        img.data[i] = c.0;
        img.data[i + 1] = c.1;
        img.data[i + 2] = c.2;
    }
}

/// Copy `src` into `canvas` at horizontal pixel offset `x_off`, one whole
/// row slice at a time (3 bytes/px contiguous) instead of per-pixel.
fn blit(canvas: &mut RgbImage, src: &RgbImage, x_off: u32) {
    let cw = canvas.width as usize;
    let sw = src.width as usize;
    for y in 0..src.height as usize {
        let dst = (y * cw + x_off as usize) * 3;
        canvas.data[dst..dst + sw * 3]
            .copy_from_slice(&src.data[y * sw * 3..(y + 1) * sw * 3]);
    }
}

fn draw_circle(img: &mut RgbImage, cx: i32, cy: i32, r: i32, c: (u8, u8, u8)) {
    for deg in (0..360).step_by(4) {
        let rad = (deg as f32).to_radians();
        put_px(img, cx + (rad.cos() * r as f32) as i32, cy + (rad.sin() * r as f32) as i32, c);
    }
}

/// Bresenham line.
fn draw_line(img: &mut RgbImage, x0: i32, y0: i32, x1: i32, y1: i32, c: (u8, u8, u8)) {
    let (mut x0, mut y0) = (x0, y0);
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        put_px(img, x0, y0, c);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

/// Match line colors cycle through a small palette.
const COLORS: [(u8, u8, u8); 6] = [
    (0, 255, 0),   // green
    (0, 200, 255), // cyan-ish
    (255, 128, 0), // orange
    (255, 0, 255), // magenta
    (255, 255, 0), // yellow
    (128, 255, 128),
];

/// Build the side-by-side visualization image.
pub fn draw_matches(
    a: &RgbImage,
    b: &RgbImage,
    da: &DescriptorSet,
    db: &DescriptorSet,
    matches: &[Match],
) -> RgbImage {
    let w = a.width + b.width;
    let h = a.height.max(b.height);
    let mut canvas = RgbImage::new(w, h, vec![0u8; (w * h * 3) as usize]);
    // blit images
    blit(&mut canvas, a, 0);
    blit(&mut canvas, b, a.width);
    for (i, m) in matches.iter().enumerate() {
        let c = COLORS[i % COLORS.len()];
        let ka = da.keypoints[m.a_idx];
        let kb = db.keypoints[m.b_idx];
        draw_circle(&mut canvas, ka.x as i32, ka.y as i32, 3, c);
        draw_circle(&mut canvas, kb.x as i32 + a.width as i32, kb.y as i32, 3, c);
        draw_line(
            &mut canvas,
            ka.x as i32,
            ka.y as i32,
            kb.x as i32 + a.width as i32,
            kb.y as i32,
            c,
        );
    }
    canvas
}
