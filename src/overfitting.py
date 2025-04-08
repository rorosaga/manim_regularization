from manim import *
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

class OverfittingDemo(Scene):
    def construct(self):
        # Create axes without coordinates (to avoid LaTeX dependency)
        axes = Axes(
            x_range=[-1, 11, 1],
            y_range=[-1, 11, 1],
            axis_config={"include_tip": False},
        )
        
        # Create custom text labels instead of using get_axis_labels
        x_label = Text("x", font="Arial").next_to(axes.x_axis.get_end(), RIGHT)
        y_label = Text("y", font="Arial").next_to(axes.y_axis.get_end(), UP)
        axes_labels = VGroup(x_label, y_label)
        
        # Generate random data points with some noise around a simple function
        np.random.seed(42)  # For reproducibility
        x = np.linspace(0, 10, 10)
        y = 1 + 2 * x + np.random.normal(0, 2, 10)  # Linear function with noise
        
        # Create dots for the scatter plot
        dots = VGroup(*[Dot(axes.c2p(x_val, y_val), color=BLUE) 
                       for x_val, y_val in zip(x, y)])
        
        # Add elements to the scene
        self.play(
            Create(axes),
            Write(axes_labels),
            FadeIn(dots)
        )
        self.wait()
        
        # Fit and show polynomials of increasing degrees
        X = x.reshape(-1, 1)
        
        # Use Text with font that doesn't require LaTeX
        degree_text = Text("Polynomial Degree: 1", font="Arial", font_size=36).to_corner(UL)
        self.play(Write(degree_text))
        
        # Initial linear fit
        line = self.get_regression_curve(axes, X, y, 1)
        self.play(Create(line))
        self.wait()
        
        # Show increasing polynomial degrees
        max_degree = 9
        for degree in range(2, max_degree + 1):
            new_line = self.get_regression_curve(axes, X, y, degree, color=RED if degree > 5 else GREEN)
            new_degree_text = Text(f"Polynomial Degree: {degree}", font="Arial", font_size=36).to_corner(UL)
            
            self.play(
                ReplacementTransform(line, new_line),
                ReplacementTransform(degree_text, new_degree_text)
            )
            line = new_line
            degree_text = new_degree_text
            self.wait()
        
        # Add a title at the end
        title = Text("Overfitting in Polynomial Regression", font="Arial", font_size=42).to_edge(UP)
        self.play(Write(title))
        self.wait(2)
    
    def get_regression_curve(self, axes, X, y, degree, color=YELLOW, num_points=100):
        # Fit polynomial regression
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X, y)
        
        # Generate points for the curve
        x_curve = np.linspace(0, 10, num_points).reshape(-1, 1)
        y_curve = model.predict(x_curve)
        
        # Create the curve
        curve_points = [axes.c2p(x_val[0], y_val) for x_val, y_val in zip(x_curve, y_curve)]
        curve = VMobject(color=color)
        curve.set_points_smoothly(curve_points)
        
        return curve

if __name__ == "__main__":
    scene = OverfittingDemo()
    scene.render()
