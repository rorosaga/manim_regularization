# How do we know when our model is overfitting? Show the graph of loss x epochs and how as the ephocs increase, the loss decreases, but there is a point where the validation loss starts to increase again.

from manim import *
import numpy as np

# Add config class for 1080p resolution
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_height = 8
config.frame_width = 14.22
config.media_dir = "C:/manim_output"

class LossAnimation(Scene):
    def construct(self):
        # Create axes
        axes = Axes(
            x_range=[0, 100, 20],
            y_range=[0, 1, 0.2],
            x_length=10,
            y_length=6,
            axis_config={"include_tip": True},
        )
        
        # Add labels
        x_label = Text("Epochs", font_size=30).next_to(axes.x_axis.get_end(), RIGHT)
        y_label = Text("Loss", font_size=30).next_to(axes.y_axis.get_end(), UP)
        axes_labels = VGroup(x_label, y_label)
        
        # Create loss curves functions
        def training_loss(x):
            return 0.8 * np.exp(-0.03 * x) + 0.05
        
        def validation_loss(x):
            return 0.8 * np.exp(-0.025 * x) + 0.1 + 0.001 * max(0, x - 40)**1.5
        
        # Create the curves
        train_curve = axes.plot(
            training_loss,
            x_range=[0, 100],
            color=BLUE
        )
        
        val_curve = axes.plot(
            validation_loss,
            x_range=[0, 100],
            color=RED
        )
        
        # Add curve labels
        train_label = Text("Training Loss", font_size=24, color=BLUE)
        train_label.next_to(axes.c2p(95, training_loss(95)), UP + RIGHT)
        
        val_label = Text("Validation Loss", font_size=24, color=RED)
        val_label.next_to(axes.c2p(95, validation_loss(95)), DOWN + RIGHT)
        
        # Add vertical line for tracking progress
        tracked_point = ValueTracker(0)
        vertical_line = always_redraw(
            lambda: axes.get_vertical_line(
                axes.i2gp(tracked_point.get_value(), train_curve)
            ).set_color(YELLOW)
        )
        
        # Add epoch counter
        epoch_counter = always_redraw(
            lambda: Text(f"Epoch: {int(tracked_point.get_value())}", font_size=36)
            .to_edge(UP)
        )
        
        # Add the best model indicator - where validation loss is minimum
        best_epoch = 40  # The approximate minimum of validation loss
        best_point = axes.c2p(best_epoch, validation_loss(best_epoch))
        best_indicator = Cross(stroke_width=6).scale(0.5).set_color(GREEN)
        best_indicator.move_to(best_point)
        best_label = Text("Best Model", font_size=24, color=GREEN)
        best_label.next_to(best_point, UP + LEFT)
        
        # Add elements to scene
        self.add(axes, axes_labels, train_curve, val_curve, 
                 train_label, val_label, vertical_line, 
                 epoch_counter)
        
        # Animate the progress of training
        self.play(tracked_point.animate.set_value(100), run_time=8, rate_func=linear)
        
        # Add best model marker
        # self.play(Create(best_indicator), Write(best_label))
        
        # Final pause to observe
        # self.wait(2)
        
        # Show overfitting zone - Fix using fill_between instead of get_area
        def create_overfitting_zone():
            xs = np.linspace(best_epoch, 100, 100)
            points = [axes.c2p(x, validation_loss(x)) for x in xs]
            baseline_points = [axes.c2p(x, 0) for x in xs]
            
            # Create a polygon with the points
            vertices = points + baseline_points[::-1]
            overfitting_zone = Polygon(*vertices, color=RED_A, fill_opacity=0.3, stroke_width=0)
            return overfitting_zone

        # Replace the problematic code section
        overfitting_zone = create_overfitting_zone()
        overfitting_label = Text("Overfitting", font_size=36, color=RED)
        overfitting_label.move_to(axes.c2p(85, 0.3))
        
        self.play(FadeIn(overfitting_zone), Write(overfitting_label))
        
        # Final pause
        self.wait(3)


# Run the animation
if __name__ == "__main__":
    scene = LossAnimation()
    scene.render()
