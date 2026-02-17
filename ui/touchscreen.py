#!/usr/bin/env python3
"""
Touchscreen UI for AI Rover.
Tap to select target → confirm with metric depth → drive precise distance.
"""
import pygame
import numpy as np
import time
import threading
import os
from dataclasses import dataclass
from typing import Optional
from enum import Enum

os.environ['DISPLAY'] = ':0'
os.environ['SDL_FBDEV'] = '/dev/fb0'

from vision.depth import DepthEstimator, DepthFrame
from track_controller import TrackController


class RoverState(Enum):
    IDLE = "IDLE"
    CONFIRMING = "CONFIRM?"
    MEASURING = "MEASURING"
    TURNING = "TURNING"
    DRIVING = "DRIVING"
    ARRIVED = "ARRIVED"


@dataclass 
class NavTarget:
    """Navigation target from screen tap."""
    screen_x: int
    screen_y: int
    depth_x: int
    depth_y: int
    angle: float
    distance_mm: Optional[float] = None
    timestamp: float = 0.0


class RoverUI:
    """Touchscreen interface for rover control."""
    
    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (50, 50, 55)
    DARK_GRAY = (25, 25, 30)
    RED = (220, 70, 70)
    GREEN = (70, 200, 100)
    BLUE = (70, 130, 220)
    YELLOW = (230, 210, 70)
    ORANGE = (255, 150, 50)
    CYAN = (70, 200, 220)
    
    # Speeds
    DRIVE_SPEED = 600
    TURN_SPEED = 350
    
    def __init__(self, fullscreen: bool = True, screen_size: tuple[int, int] = (800, 480)):
        print("Initializing Rover UI...")
        
        pygame.init()
        pygame.mouse.set_visible(False)
        
        if fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.width, self.height = self.screen.get_size()
        else:
            self.width, self.height = screen_size
            self.screen = pygame.display.set_mode(screen_size)
        
        pygame.display.set_caption("AI Rover")
        
        # Fonts
        self.font_title = pygame.font.Font(None, 42)
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 24)
        self.font_big = pygame.font.Font(None, 52)
        
        # Layout
        self.header_height = 50
        self.panel_width = 280
        self.feed_width = self.width - self.panel_width
        self.feed_height = self.height - self.header_height
        
        self.feed_rect = pygame.Rect(0, self.header_height, self.feed_width, self.feed_height)
        self.panel_rect = pygame.Rect(self.feed_width, self.header_height, self.panel_width, self.feed_height)
        
        # Buttons
        btn_w, btn_h = 110, 50
        btn_y = self.height - btn_h - 20
        self.btn_stop = pygame.Rect(self.feed_width + 15, btn_y, btn_w, btn_h)
        self.btn_home = pygame.Rect(self.feed_width + btn_w + 30, btn_y, btn_w, btn_h)
        
        # Confirm dialog buttons (centered on feed)
        confirm_w, confirm_h = 120, 60
        confirm_y = self.feed_rect.centery + 60
        self.btn_confirm_go = pygame.Rect(
            self.feed_rect.centerx - confirm_w - 20, confirm_y, confirm_w, confirm_h
        )
        self.btn_confirm_cancel = pygame.Rect(
            self.feed_rect.centerx + 20, confirm_y, confirm_w, confirm_h
        )
        
        # State
        self.state = RoverState.IDLE
        self.target: Optional[NavTarget] = None
        self.running = False
        self.latest_frame: Optional[DepthFrame] = None
        
        # Nav metrics
        self.nav_start_time = 0.0
        self.nav_start_pos = (0.0, 0.0)
        self.status_message = "Tap camera to set target"
        
        # Hardware
        print("  Starting vision...")
        self.depth = DepthEstimator()
        print("  Starting controller...")
        self.controller = TrackController()
        
        # Enable metric depth
        print("  Enabling metric depth...")
        self.depth.enable_metric_depth(calibration_interval=5.0)
        
        # Nav thread
        self.nav_thread: Optional[threading.Thread] = None
        self.nav_cancel = threading.Event()
        
        print(f"UI ready: {self.width}x{self.height}")
    
    def start(self):
        """Start systems and UI loop."""
        self.depth.start()
        self.controller.start()
        self.running = True
        
        # Warm up
        time.sleep(0.3)
        for _ in range(5):
            self.depth.capture()
        
        self._run_loop()
    
    def stop(self):
        """Stop everything."""
        self.running = False
        self.nav_cancel.set()
        if self.nav_thread and self.nav_thread.is_alive():
            self.nav_thread.join(timeout=1.0)
        self.controller.stop()
        self.controller.shutdown()
        self.depth.stop()
        pygame.quit()
    
    def _handle_tap(self, pos: tuple[int, int]):
        """Handle screen tap."""
        x, y = pos
        
        # Always allow stop
        if self.btn_stop.collidepoint(x, y):
            self._stop_navigation()
            return
        
        if self.btn_home.collidepoint(x, y):
            self._return_home()
            return
        
        # Confirmation dialog buttons
        if self.state == RoverState.CONFIRMING:
            if self.btn_confirm_go.collidepoint(x, y):
                self._execute_navigation()
                return
            elif self.btn_confirm_cancel.collidepoint(x, y):
                self.state = RoverState.IDLE
                self.target = None
                self.status_message = "Cancelled"
                return
            # Tap elsewhere cancels
            elif not self.feed_rect.collidepoint(x, y):
                return
        
        # New target selection (only when idle or confirming)
        if self.state in [RoverState.IDLE, RoverState.CONFIRMING] and self.feed_rect.collidepoint(x, y):
            self._select_target(x, y)
    
    def _select_target(self, tap_x: int, tap_y: int):
        """Select target and measure distance."""
        if self.latest_frame is None:
            return
        
        self.state = RoverState.MEASURING
        self.status_message = "Measuring distance..."
        
        # Convert tap to depth coordinates
        rel_x = (tap_x - self.feed_rect.x) / self.feed_rect.width
        rel_y = (tap_y - self.feed_rect.y) / self.feed_rect.height
        
        depth_h, depth_w = self.latest_frame.normalized.shape
        depth_x = int(rel_x * depth_w)
        depth_y = int(rel_y * depth_h)
        depth_x = max(5, min(depth_w - 5, depth_x))
        depth_y = max(5, min(depth_h - 5, depth_y))
        
        # Calculate angle
        angle = (rel_x - 0.5) * 80  # -40 to +40 degrees
        
        self.target = NavTarget(
            screen_x=tap_x,
            screen_y=tap_y,
            depth_x=depth_x,
            depth_y=depth_y,
            angle=angle,
            timestamp=time.time()
        )
        
        # Force a fresh metric depth reading
        self._measure_target_distance()
    
    def _measure_target_distance(self):
        """Get metric distance to target using Depth Anything V2."""
        if self.target is None:
            return
        
        # Force metric recalibration by resetting timer
        self.depth._last_calibration = 0
        
        # Capture fresh frame and run metric depth
        frame = self.depth.capture()
        
        # Wait a moment for metric calibration to run
        time.sleep(0.1)
        frame = self.depth.capture()
        
        if frame.metric is not None:
            # Sample region around tap point
            margin = 15
            y1 = max(0, self.target.depth_y - margin)
            y2 = min(frame.metric.shape[0], self.target.depth_y + margin)
            x1 = max(0, self.target.depth_x - margin)
            x2 = min(frame.metric.shape[1], self.target.depth_x + margin)
            
            region = frame.metric[y1:y2, x1:x2]
            self.target.distance_mm = float(np.median(region))
            
            self.status_message = f"Target: {self.target.distance_mm:.0f}mm at {self.target.angle:+.0f}°"
        else:
            # Fallback - estimate from normalized
            self.target.distance_mm = None
            self.status_message = f"Target: ~??? mm at {self.target.angle:+.0f}°"
        
        self.state = RoverState.CONFIRMING
    
    def _execute_navigation(self):
        """Execute the confirmed navigation."""
        if self.target is None:
            return
        
        self.nav_start_time = time.time()
        self.nav_start_pos = self.controller.position
        
        self.nav_cancel.clear()
        self.nav_thread = threading.Thread(target=self._navigate, daemon=True)
        self.nav_thread.start()
    
    def _navigate(self):
        """Navigation: turn exact angle, drive exact distance. That's it."""
        if self.target is None:
            return
        
        try:
            # === Step 1: Turn ===
            if abs(self.target.angle) > 3:
                self.state = RoverState.TURNING
                self.status_message = f"Turning {self.target.angle:+.0f}°"
                print(f"[NAV] Turn {self.target.angle:.1f}°")
                
                self.controller.pivot_turn(self.target.angle, speed_tps=350)
                self.controller.wait_for_turn()
                
                if self.nav_cancel.is_set():
                    self.controller.stop()
                    self.state = RoverState.IDLE
                    return
                
                print(f"[NAV] Turn done: {self.controller.heading:.1f}°")
                time.sleep(0.2)
            
            # === Step 2: Drive ===
            if self.target.distance_mm and self.target.distance_mm > 200:
                self.state = RoverState.DRIVING
                drive_dist = self.target.distance_mm - 150  # Stop 15cm short
                
                self.status_message = f"Driving {drive_dist:.0f}mm"
                print(f"[NAV] Drive {drive_dist:.0f}mm")
                
                self.controller.drive_distance(drive_dist, speed_tps=600)
                
                print(f"[NAV] Done. Pos: {self.controller.position}")
                self.status_message = f"Arrived!"
            
            self.state = RoverState.ARRIVED
            
        except Exception as e:
            self.controller.stop()
            self.state = RoverState.IDLE
            self.status_message = f"Error: {e}"
            print(f"[NAV] Error: {e}")
    
    def _stop_navigation(self):
        """Stop navigation."""
        self.nav_cancel.set()
        self.controller.stop()
        self.state = RoverState.IDLE
        self.target = None
        self.status_message = "Stopped"
    
    def _return_home(self):
        """Return to start position."""
        self._stop_navigation()
        
        x, y = self.controller.position
        if abs(x) < 50 and abs(y) < 50:
            self.status_message = "Already home"
            return
        
        self.nav_start_time = time.time()
        self.nav_start_pos = (x, y)
        
        self.nav_cancel.clear()
        self.nav_thread = threading.Thread(target=self._execute_home, daemon=True)
        self.nav_thread.start()
    
    def _execute_home(self):
        """Return home using TrackController."""
        x, y = self.controller.position
        distance = np.sqrt(x**2 + y**2)
        
        angle_to_home = np.degrees(np.arctan2(-y, -x)) - self.controller.heading
        while angle_to_home > 180: angle_to_home -= 360
        while angle_to_home < -180: angle_to_home += 360
        
        if abs(angle_to_home) > 5:
            self.state = RoverState.TURNING
            self.status_message = f"Turning home {angle_to_home:+.0f}°"
            
            self.controller.pivot_turn(angle_to_home, speed_tps=self.TURN_SPEED)
            while self.controller.is_turning and not self.nav_cancel.is_set():
                time.sleep(0.02)
        
        if self.nav_cancel.is_set():
            self.state = RoverState.IDLE
            return
        
        self.state = RoverState.DRIVING
        self.status_message = f"Driving home {distance:.0f}mm"
        
        self.controller.drive_distance(distance, speed_tps=self.DRIVE_SPEED)
        
        self.state = RoverState.ARRIVED
        self.status_message = "Home!"
    
    def _distance_traveled(self) -> float:
        x, y = self.controller.position
        sx, sy = self.nav_start_pos
        return np.sqrt((x - sx)**2 + (y - sy)**2)
    
    def _draw(self):
        """Draw UI."""
        self.screen.fill(self.DARK_GRAY)
        
        # Header
        pygame.draw.rect(self.screen, self.GRAY, (0, 0, self.width, self.header_height))
        title = self.font_title.render("AI ROVER", True, self.WHITE)
        self.screen.blit(title, (20, 10))
        
        # State badge
        colors = {
            RoverState.IDLE: self.GRAY,
            RoverState.CONFIRMING: self.YELLOW,
            RoverState.MEASURING: self.ORANGE,
            RoverState.TURNING: self.YELLOW,
            RoverState.DRIVING: self.GREEN,
            RoverState.ARRIVED: self.CYAN
        }
        badge = self.font_medium.render(self.state.value, True, self.WHITE)
        bw = badge.get_width() + 24
        br = pygame.Rect(self.width - bw - 15, 10, bw, 30)
        pygame.draw.rect(self.screen, colors[self.state], br, border_radius=15)
        self.screen.blit(badge, (br.x + 12, br.y + 5))
        
        # Camera feed
        if self.latest_frame is not None:
            surf = self._array_to_surface(self.latest_frame.rgb, (self.feed_rect.width, self.feed_rect.height))
            self.screen.blit(surf, self.feed_rect.topleft)
            
            # Center crosshair
            cx, cy = self.feed_rect.centerx, self.feed_rect.centery
            pygame.draw.circle(self.screen, (255, 255, 255, 128), (cx, cy), 30, 1)
            pygame.draw.line(self.screen, self.WHITE, (cx - 20, cy), (cx + 20, cy), 1)
            pygame.draw.line(self.screen, self.WHITE, (cx, cy - 20), (cx, cy + 20), 1)
            
            # Target marker
            if self.target:
                t = time.time()
                r = int(12 + 4 * np.sin(t * 6))
                
                # Draw at original tap position
                tx, ty = self.target.screen_x, self.target.screen_y
                pygame.draw.circle(self.screen, self.GREEN, (tx, ty), r + 6, 3)
                pygame.draw.circle(self.screen, self.WHITE, (tx, ty), r, 2)
                pygame.draw.line(self.screen, self.GREEN, (tx - 20, ty), (tx + 20, ty), 2)
                pygame.draw.line(self.screen, self.GREEN, (tx, ty - 20), (tx, ty + 20), 2)
                
                # Distance label at target
                if self.target.distance_mm:
                    dist_label = self.font_medium.render(f"{self.target.distance_mm:.0f}mm", True, self.GREEN)
                    self.screen.blit(dist_label, (tx + 15, ty - 25))
        
        # Confirmation overlay
        if self.state == RoverState.CONFIRMING and self.target:
            # Semi-transparent overlay
            overlay = pygame.Surface((self.feed_rect.width, self.feed_rect.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            self.screen.blit(overlay, self.feed_rect.topleft)
            
            # Info box
            info_rect = pygame.Rect(self.feed_rect.centerx - 150, self.feed_rect.centery - 80, 300, 100)
            pygame.draw.rect(self.screen, self.GRAY, info_rect, border_radius=10)
            pygame.draw.rect(self.screen, self.WHITE, info_rect, 2, border_radius=10)
            
            # Target info
            if self.target.distance_mm:
                dist_text = self.font_big.render(f"{self.target.distance_mm:.0f} mm", True, self.WHITE)
            else:
                dist_text = self.font_big.render("??? mm", True, self.ORANGE)
            dist_rect = dist_text.get_rect(center=(info_rect.centerx, info_rect.centery - 15))
            self.screen.blit(dist_text, dist_rect)
            
            angle_text = self.font_medium.render(f"Turn: {self.target.angle:+.0f}°", True, (180, 180, 180))
            angle_rect = angle_text.get_rect(center=(info_rect.centerx, info_rect.centery + 25))
            self.screen.blit(angle_text, angle_rect)
            
            # Confirm/Cancel buttons
            self._button(self.btn_confirm_go, "GO", self.GREEN)
            self._button(self.btn_confirm_cancel, "CANCEL", self.RED)
        
        # Feed label
        if self.state == RoverState.IDLE:
            lbl = self.font_small.render("TAP TO SET TARGET", True, (150, 150, 150))
            self.screen.blit(lbl, (self.feed_rect.x + 10, self.feed_rect.y + 10))
        
        # Side panel
        pygame.draw.rect(self.screen, self.GRAY, self.panel_rect)
        px = self.panel_rect.x + 15
        py = self.panel_rect.y + 15
        
        # Status
        self._section(px, py, "STATUS")
        py += 30
        
        # Wrap status message if needed
        status = self.font_medium.render(self.status_message[:25], True, self.WHITE)
        self.screen.blit(status, (px, py))
        py += 35
        
        # Navigation
        self._section(px, py, "NAVIGATION")
        py += 30
        
        if self.target and self.target.distance_mm:
            self._metric(px, py, "Target", f"{self.target.distance_mm:.0f} mm")
            py += 26
            self._metric(px, py, "Angle", f"{self.target.angle:+.1f}°")
            py += 26
        
        elapsed = time.time() - self.nav_start_time if self.nav_start_time else 0
        if self.state in [RoverState.DRIVING, RoverState.TURNING]:
            self._metric(px, py, "Time", f"{elapsed:.1f} s")
            py += 26
        
        py += 10
        
        # Position
        self._section(px, py, "POSITION")
        py += 30
        
        x, y = self.controller.position
        hdg = self.controller.heading
        
        self._metric(px, py, "X", f"{x:.0f} mm")
        py += 26
        self._metric(px, py, "Y", f"{y:.0f} mm")
        py += 26
        self._metric(px, py, "Heading", f"{hdg:+.1f}°")
        py += 35
        
        # Depth preview
        if self.latest_frame is not None:
            self._section(px, py, "DEPTH")
            py += 25
            dw, dh = self.panel_width - 30, 60
            ds = self._array_to_surface(self.latest_frame.colorized, (dw, dh))
            self.screen.blit(ds, (px, py))
        
        # Buttons
        stop_color = self.RED if self.state != RoverState.IDLE else (100, 50, 50)
        self._button(self.btn_stop, "STOP", stop_color)
        self._button(self.btn_home, "HOME", self.BLUE)
        
        pygame.display.flip()
    
    def _section(self, x, y, text):
        lbl = self.font_small.render(text, True, (120, 120, 120))
        self.screen.blit(lbl, (x, y))
        pygame.draw.line(self.screen, (70, 70, 70), (x, y + 18), (x + self.panel_width - 40, y + 18), 1)
    
    def _metric(self, x, y, label, value):
        l = self.font_small.render(label, True, (140, 140, 140))
        v = self.font_medium.render(value, True, self.WHITE)
        self.screen.blit(l, (x, y))
        self.screen.blit(v, (x + 90, y - 2))
    
    def _button(self, rect, text, color):
        pygame.draw.rect(self.screen, color, rect, border_radius=10)
        pygame.draw.rect(self.screen, self.WHITE, rect, 2, border_radius=10)
        lbl = self.font_large.render(text, True, self.WHITE)
        self.screen.blit(lbl, lbl.get_rect(center=rect.center))
    
    def _array_to_surface(self, arr, size):
        if len(arr.shape) == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1]
        surf = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        return pygame.transform.scale(surf, size)
    
    def _run_loop(self):
        clock = pygame.time.Clock()
        try:
            while self.running:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        self.running = False
                    elif ev.type == pygame.KEYDOWN and ev.key in (pygame.K_q, pygame.K_ESCAPE):
                        self.running = False
                    elif ev.type == pygame.MOUSEBUTTONDOWN:
                        self._handle_tap(ev.pos)
                    elif ev.type == pygame.FINGERDOWN:
                        self._handle_tap((int(ev.x * self.width), int(ev.y * self.height)))
                
                # Only capture frames when not measuring (metric depth blocks)
                if self.state != RoverState.MEASURING:
                    self.latest_frame = self.depth.capture()
                
                self._draw()
                clock.tick(30)
        finally:
            self.stop()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--windowed', action='store_true')
    args = p.parse_args()
    
    ui = RoverUI(fullscreen=not args.windowed)
    try:
        ui.start()
    except KeyboardInterrupt:
        pass
    finally:
        ui.stop()


if __name__ == '__main__':
    main()