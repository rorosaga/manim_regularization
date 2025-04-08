# Show a scatterplot of points resemmbling a curve with a polynomial regression model of degree 1. As the degrees increase, the model starts to fit the noise in the data until it overfits.

from manim import *
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Add config class for 1080p resolution
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_height = 8
config.frame_width = 14.22

class OverfittingAnimation(Scene):
    def construct(self):
        # Generate data
        np.random.seed(42)
        x = np.linspace(-1, 1, 20).reshape(-1, 1)
        true_func = lambda x: np.sin(np.pi * x).flatten()
        y = true_func(x) + np.random.normal(0, 0.25, size=x.shape[0])
        x_plot = np.linspace(-1, 1, 100).reshape(-1, 1)
        
        # Create axes
        axes = Axes(
            x_range=[-1.2, 1.2, 0.5],
            y_range=[-1.5, 1.5, 0.5],
            axis_config={"include_tip": False},
        ).scale(0.7)
        
        # Add labels - use Text instead of TeX
        x_label = Text("x", font_size=24).next_to(axes.x_axis.get_end(), RIGHT)
        y_label = Text("y", font_size=24).next_to(axes.y_axis.get_end(), UP)
        axes_labels = VGroup(x_label, y_label)
        
        # Plot scatter points
        dots = VGroup(*[
            Dot(axes.coords_to_point(x_i[0], y_i), radius=0.05, color=BLUE)
            for x_i, y_i in zip(x, y)
        ])
        
        # Add true function curve
        # true_curve = axes.plot(
        #     lambda x: np.sin(np.pi * x),
        #     x_range=[-1, 1],
        #     color=GREEN
        # )
        # true_curve_label = Text("", font_size=20).set_color(GREEN)
        # true_curve_label.next_to(axes, UP).shift(LEFT * 3)
        
        # Create degree counter
        # degree_counter = Text("Degree: 0", font_size=24)
        # degree_counter.to_corner(UR).shift(LEFT * 0.5)
        
        # Add elements to scene
        self.add(axes, axes_labels, dots)
        
        # Initial polynomial degree 1 (linear)
        max_degree = 12
        current_curve = None
        
        for degree in range(1, max_degree + 1):
            # Train polynomial model
            model = make_pipeline(
                PolynomialFeatures(degree),
                LinearRegression()
            )
            model.fit(x, y)
            y_pred = model.predict(x_plot)
            
            # Create new curve
            new_curve = axes.plot_line_graph(
                x_plot.flatten(), y_pred, line_color=RED,
                add_vertex_dots=False
            )
            
            # Update degree counter
            new_counter = Text(f"Degree: {degree}", font_size=24)
            new_counter.to_corner(UR).shift(LEFT * 0.5)
            
            # Animate the change
            if current_curve is None:
                self.play(Create(new_curve), Write(new_counter), run_time=1)
            else:
                self.play(
                    ReplacementTransform(current_curve, new_curve),
                    ReplacementTransform(degree_counter, new_counter),
                    run_time=1
                )
            
            # Update current curve and counter
            current_curve = new_curve
            degree_counter = new_counter
            
            # Pause slightly longer for higher degrees to emphasize overfitting
            if degree >= 10:
                self.wait(0.5)
        
        # Final pause to observe complete overfitting
        self.wait(2)
