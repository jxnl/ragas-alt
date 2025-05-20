from setuptools import setup, find_packages

setup(
    name="ragas-alt",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Add your project dependencies here
        # e.g., "numpy>=1.0",
    ],
    # Add other metadata like author, description, etc.
    author="Your Name",
    author_email="your.email@example.com",
    description="A short description of your project.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="YOUR_PROJECT_URL",  # Replace with your project's URL
)
