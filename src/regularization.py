from manim import *
import numpy as np

class RegularizationExplanation(Scene):
    def construct(self):
        # Set up the custom output directory
        config.media_dir = "media"
        config.output_dir = "regularization1"
        
        # Introduction
        self.introduction()
        
        # Explain the problem of overfitting
        self.explain_overfitting()
        
        # Introduce the concept of regularization
        self.introduce_regularization()
        
        # Explain L2 regularization (Ridge)
        self.explain_l2_regularization()
        
        # Explain L1 regularization (Lasso)
        self.explain_l1_regularization()
        
        # Compare different regularization techniques
        self.compare_regularization()
        
        # Conclusion
        self.conclusion()
    
    def introduction(self):
        title = Text("Regularization in Machine Learning", font_size=40).to_edge(UP)
        subtitle = Text("A Step-by-Step Explanation", font_size=30).next_to(title, DOWN)
        
        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle), run_time=1)
        self.wait(2)
        
        what_is = Text("What is Regularization?", font_size=36)
        definition = Text(
            "Regularization is a technique to prevent overfitting\n"
            "by adding a penalty term to the loss function.",
            font_size=24,
            line_spacing=1.5
        ).next_to(what_is, DOWN, buff=0.5)
        
        self.play(
            FadeOut(subtitle),
            Transform(title, what_is)
        )
        self.play(Write(definition), run_time=2)
        self.wait(3)
        
        self.play(FadeOut(definition), FadeOut(title))
    
    def explain_overfitting(self):
        title = Text("The Problem: Overfitting", font_size=36).to_edge(UP)
        self.play(Write(title))
        
        # Create axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 2, 1],
            axis_config={"include_tip": False},
            x_length=10,
            y_length=6
        ).shift(DOWN * 0.5)
        
        x_label = Text("x", font_size=24).next_to(axes.x_axis.get_end(), RIGHT)
        y_label = Text("y", font_size=24).next_to(axes.y_axis.get_end(), UP)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Generate training data (with noise)
        np.random.seed(42)  # For reproducibility
        x_values = np.linspace(-2.5, 2.5, 12)
        y_values = 0.5 * x_values + 0.3 * np.random.randn(len(x_values))
        
        # Plot training data points
        dots = VGroup(*[
            Dot(axes.c2p(x, y), color=BLUE, radius=0.08)
            for x, y in zip(x_values, y_values)
        ])
        
        data_label = Text("Training Data", font_size=24, color=BLUE).next_to(axes, UP).shift(LEFT * 3)
        
        self.play(FadeIn(dots), Write(data_label))
        self.wait(1)
        
        # Simple linear model (good fit)
        linear_model = axes.plot(lambda x: 0.5 * x, color=GREEN)
        linear_label = Text("Simple Model", font_size=24, color=GREEN).next_to(axes, UP).shift(RIGHT * 3)
        
        self.play(Create(linear_model), Write(linear_label))
        self.wait(2)
        
        # Complex model (overfitting)
        def complex_model(x):
            return 0.5 * x + 0.3 * np.sin(5 * x) + 0.2 * np.cos(3 * x)
        
        complex_curve = axes.plot(complex_model, color=RED)
        complex_label = Text("Complex Model (Overfitting)", font_size=24, color=RED).next_to(linear_label, DOWN)
        
        self.play(Create(complex_curve), Write(complex_label))
        self.wait(2)
        
        # Add explanation about overfitting
        overfitting_box = Rectangle(height=3, width=6, color=YELLOW, fill_opacity=0.2).to_edge(DOWN)
        overfitting_text = Text(
            "Overfitting: Model learns noise in the data\n"
            "rather than the underlying pattern.\n"
            "Performs well on training data but poorly on new data.",
            font_size=20,
            line_spacing=1.2
        ).move_to(overfitting_box.get_center())
        
        self.play(
            Create(overfitting_box),
            Write(overfitting_text)
        )
        self.wait(3)
        
        self.play(
            FadeOut(overfitting_box), 
            FadeOut(overfitting_text),
            FadeOut(dots),
            FadeOut(linear_model),
            FadeOut(complex_curve),
            FadeOut(linear_label),
            FadeOut(complex_label),
            FadeOut(data_label),
            FadeOut(axes),
            FadeOut(x_label),
            FadeOut(y_label),
            FadeOut(title)
        )
    
    def introduce_regularization(self):
        title = Text("Introducing Regularization", font_size=36).to_edge(UP)
        self.play(Write(title))
        
        # Create a basic loss function illustration
        equation_group = VGroup()
        
        loss_text = Text("Standard Loss:", font_size=28).shift(UP * 1.5)
        loss_eq = Text("L(θ) = (1/n)∑(yi - ŷi)²", font_size=32)
        loss_eq_text = Text("Mean Squared Error", font_size=20, color=GRAY).next_to(loss_eq, DOWN)
        
        equation_group.add(loss_text, loss_eq, loss_eq_text)
        
        reg_text = Text("Regularized Loss:", font_size=28).shift(DOWN * 1)
        reg_eq = Text("Lreg(θ) = L(θ) + λ · Penalty(θ)", font_size=32)
        reg_eq_text = Text("Loss + Regularization Term", font_size=20, color=GRAY).next_to(reg_eq, DOWN)
        
        equation_group.add(reg_text, reg_eq, reg_eq_text)
        
        self.play(Write(loss_text))
        self.play(Write(loss_eq), Write(loss_eq_text))
        self.wait(2)
        
        self.play(Write(reg_text))
        self.play(Write(reg_eq), Write(reg_eq_text))
        self.wait(2)
        
        # Highlight the regularization parameter
        lambda_highlight = SurroundingRectangle(reg_eq[13:14], color=YELLOW)
        lambda_text = Text("Regularization Strength (λ)", font_size=20, color=YELLOW).next_to(lambda_highlight, DOWN)
        
        self.play(Create(lambda_highlight), Write(lambda_text))
        self.wait(2)
        
        self.play(
            FadeOut(equation_group),
            FadeOut(lambda_highlight),
            FadeOut(lambda_text)
        )
        
        regularization_types = VGroup(
            Text("Types of Regularization:", font_size=32),
            Text("1. L1 Regularization (Lasso)", font_size=28),
            Text("2. L2 Regularization (Ridge)", font_size=28),
            Text("3. Elastic Net (Combination of L1 and L2)", font_size=28)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5).shift(DOWN * 0.5)
        
        self.play(Write(regularization_types[0]))
        self.wait(1)
        
        for i in range(1, 4):
            self.play(FadeIn(regularization_types[i]))
            self.wait(1)
        
        self.wait(2)
        self.play(
            FadeOut(regularization_types),
            FadeOut(title)
        )
    
    def explain_l2_regularization(self):
        title = Text("L2 Regularization (Ridge)", font_size=36).to_edge(UP)
        self.play(Write(title))
        
        # L2 equation
        l2_eq = Text("L_L2(θ) = L(θ) + λ∑θj²", font_size=32)
        l2_eq.shift(UP * 1.5)
        
        # L2 regularization penalty
        l2_penalty = Text("Penalty_L2(θ) = ∑θj²", font_size=28)
        l2_penalty.next_to(l2_eq, DOWN, buff=0.5)
        
        self.play(Write(l2_eq))
        self.wait(1)
        self.play(Write(l2_penalty))
        self.wait(2)
        
        # L2 effects
        effects_title = Text("Effects:", font_size=30).shift(DOWN * 0.2)
        effects = VGroup(
            Text("• Shrinks all coefficients toward zero", font_size=24),
            Text("• Larger penalties for larger coefficients", font_size=24),
            Text("• Helps with multicollinearity", font_size=24),
            Text("• Typically improves generalization", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(effects_title, DOWN, buff=0.3)
        
        self.play(Write(effects_title))
        for effect in effects:
            self.play(FadeIn(effect))
            self.wait(0.7)
        
        self.wait(2)
        self.play(
            FadeOut(l2_eq),
            FadeOut(l2_penalty),
            FadeOut(effects_title),
            FadeOut(effects),
            FadeOut(title)
        )
    
    def explain_l1_regularization(self):
        title = Text("L1 Regularization (Lasso)", font_size=36).to_edge(UP)
        self.play(Write(title))
        
        # L1 equation
        l1_eq = Text("L_L1(θ) = L(θ) + λ∑|θj|", font_size=32)
        l1_eq.shift(UP * 1.5)
        
        # L1 regularization penalty
        l1_penalty = Text("Penalty_L1(θ) = ∑|θj|", font_size=28)
        l1_penalty.next_to(l1_eq, DOWN, buff=0.5)
        
        self.play(Write(l1_eq))
        self.wait(1)
        self.play(Write(l1_penalty))
        self.wait(2)
        
        # L1 effects
        effects_title = Text("Effects:", font_size=30).shift(DOWN * 0.2)
        effects = VGroup(
            Text("• Can reduce coefficients to exactly zero", font_size=24),
            Text("• Performs feature selection", font_size=24),
            Text("• Creates sparse models", font_size=24),
            Text("• Robust to outliers", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(effects_title, DOWN, buff=0.3)
        
        self.play(Write(effects_title))
        for effect in effects:
            self.play(FadeIn(effect))
            self.wait(0.7)
        
        self.wait(2)
        self.play(
            FadeOut(l1_eq),
            FadeOut(l1_penalty),
            FadeOut(effects_title),
            FadeOut(effects),
            FadeOut(title)
        )
    
    def compare_regularization(self):
        title = Text("Comparing Regularization Methods", font_size=36).to_edge(UP)
        self.play(Write(title))
        
        # Create axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 2, 1],
            axis_config={"include_tip": False},
            x_length=10,
            y_length=6
        ).shift(DOWN * 0.5)
        
        x_label = Text("x", font_size=24).next_to(axes.x_axis.get_end(), RIGHT)
        y_label = Text("y", font_size=24).next_to(axes.y_axis.get_end(), UP)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Generate data points
        np.random.seed(42)
        x_values = np.linspace(-2.5, 2.5, 12)
        y_values = 0.5 * x_values + 0.3 * np.random.randn(len(x_values))
        
        dots = VGroup(*[
            Dot(axes.c2p(x, y), color=BLUE, radius=0.08)
            for x, y in zip(x_values, y_values)
        ])
        
        data_label = Text("Data Points", font_size=20, color=BLUE).to_edge(RIGHT).shift(UP * 2)
        
        self.play(FadeIn(dots), Write(data_label))
        
        # Overfitted model
        def complex_model(x):
            return 0.5 * x + 0.3 * np.sin(5 * x) + 0.2 * np.cos(3 * x)
        
        overfit_curve = axes.plot(complex_model, color=RED)
        overfit_label = Text("Overfitted Model", font_size=20, color=RED).next_to(data_label, DOWN)
        
        self.play(Create(overfit_curve), Write(overfit_label))
        self.wait(1.5)
        
        # L2 regularized model
        def l2_model(x):
            return 0.48 * x + 0.12 * np.sin(5 * x) + 0.08 * np.cos(3 * x)
        
        l2_curve = axes.plot(l2_model, color=GREEN)
        l2_label = Text("L2 Regularization", font_size=20, color=GREEN).next_to(overfit_label, DOWN)
        
        self.play(Create(l2_curve), Write(l2_label))
        self.wait(1.5)
        
        # L1 regularized model (more sparse)
        def l1_model(x):
            return 0.45 * x + 0.05 * np.sin(5 * x)  # Note: Some terms reduced to zero
        
        l1_curve = axes.plot(l1_model, color=YELLOW)
        l1_label = Text("L1 Regularization", font_size=20, color=YELLOW).next_to(l2_label, DOWN)
        
        self.play(Create(l1_curve), Write(l1_label))
        self.wait(2)
        
        # Comparison text
        comparison_box = Rectangle(height=3, width=6, color=WHITE, fill_opacity=0.1).to_edge(LEFT)
        comparison_text = VGroup(
            Text("Comparison:", font_size=24),
            Text("• L2: Shrinks all coefficients", font_size=20),
            Text("• L1: Creates sparse models", font_size=20),
            Text("• Both: Prevent overfitting", font_size=20)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(comparison_box.get_center())
        
        self.play(
            Create(comparison_box),
            Write(comparison_text[0])
        )
        
        for i in range(1, 4):
            self.play(FadeIn(comparison_text[i]))
            self.wait(0.7)
        
        self.wait(3)
        
        self.play(
            FadeOut(comparison_box),
            FadeOut(comparison_text),
            FadeOut(dots),
            FadeOut(overfit_curve),
            FadeOut(l2_curve),
            FadeOut(l1_curve),
            FadeOut(data_label),
            FadeOut(overfit_label),
            FadeOut(l2_label),
            FadeOut(l1_label),
            FadeOut(axes),
            FadeOut(x_label),
            FadeOut(y_label),
            FadeOut(title)
        )
    
    def conclusion(self):
        title = Text("Key Takeaways", font_size=36).to_edge(UP)
        self.play(Write(title))
        
        takeaways = VGroup(
            Text("1. Regularization prevents overfitting", font_size=28),
            Text("2. L1 (Lasso): Creates sparse models", font_size=28),
            Text("3. L2 (Ridge): Shrinks all coefficients", font_size=28),
            Text("4. The λ parameter controls regularization strength", font_size=28),
            Text("5. Start with a small λ and increase gradually", font_size=28)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5).shift(DOWN * 0.5)
        
        for takeaway in takeaways:
            self.play(FadeIn(takeaway))
            self.wait(1)
        
        self.wait(2)
        
        thank_you = Text("Thank you for watching!", font_size=40, color=BLUE).shift(DOWN * 3)
        self.play(Write(thank_you))
        self.wait(3)
        
        self.play(
            FadeOut(title),
            FadeOut(takeaways),
            FadeOut(thank_you)
        )


if __name__ == "__main__":
    scene = RegularizationExplanation()
    scene.render()
