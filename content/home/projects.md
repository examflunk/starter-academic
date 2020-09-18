+++
# A Projects section created with the Portfolio widget.
widget = "portfolio"  # See https://sourcethemes.com/academic/docs/page-builder/
headless = true  # This file represents a page section.
active = true  # Activate this widget? true/false
weight = 65  # Order that this section will appear.

title = "Projects"
subtitle = ""

[content]
  # Page type to display. E.g. project.
  page_type = "project"
  
  # Filter toolbar (optional).
  # Add or remove as many filters (`[[content.filter_button]]` instances) as you like.
  # To show all items, set `tag` to "*".
  # To filter by a specific tag, set `tag` to an existing tag name.
  # To remove toolbar, delete/comment all instances of `[[content.filter_button]]` below.
  
  # Default filter index (e.g. 0 corresponds to the first `[[filter_button]]` instance below).
  filter_default = 0
  
   [[content.filter_button]]
     name = "All"
     tag = "*"
  
   # [[content.filter_button]]
    # name = "Acadamic Projects"
    # tag = "Deep Learning"
  
   # [[content.filter_button]]
   #  name = "Other"
   #  tag = "Demo"

# Experiences.
#   Add/remove as many `[[experience]]` blocks below as you like.
#   Required fields are `title`, `company`, and `date_start`.
#   Leave `date_end` empty if it's your current employer.
#   Begin/end multi-line descriptions with 3 quotes `"""`.
[[projects]]
  title = "Used Car Price Prediction"
  date_start = "2020-06-01"
  date_end = "2020-08-020"
  description = """This project is regarding building a model to predict used car prices for my graduate course. We took the dataset from kaggle and worked on it to build a effective model for predicting the prices."""
  
  [[projects]]
  title = "Website Development"
  date_start = "2020-01-18"
  date_end = "2020-04-020"
  description = """Developed a food Ordereing website of restaurant for its online orders and tracking sales for
the graduate course using HTML, CSS, PHP and MySQL"""
  
  [[projects]]
  title = "Fire Fighting Robot"
  date_start = "2020-01-10"
  date_end = "2020-05-25"
  description = """Built the prototype of the robot for my undergraduate course using PIC Microcontroller, Sensors, DC Motors and Drivers."""

[design]
  # Choose how many columns the section has. Valid values: 1 or 2.
  columns = "2"

  # Toggle between the various page layout types.
  #   1 = List
  #   2 = Compact
  #   3 = Card
  #   5 = Showcase
  view = 5

  # For Showcase view, flip alternate rows?
  flip_alt_rows = true

[design.background]
  # Apply a background color, gradient, or image.
  #   Uncomment (by removing `#`) an option to apply it.
  #   Choose a light or dark text color by setting `text_color_light`.
  #   Any HTML color name or Hex value is valid.
  
  # Background color.
  # color = "navy"
  
  # Background gradient.
  # gradient_start = "DeepSkyBlue"
  # gradient_end = "SkyBlue"
  
  # Background image.
  # image = "background.jpg"  # Name of image in `static/media/`.
  # image_darken = 0.6  # Darken the image? Range 0-1 where 0 is transparent and 1 is opaque.

  # Text color (true=light or false=dark).
  # text_color_light = true  
  
[advanced]
 # Custom CSS. 
 css_style = ""
 
 # CSS class.
 css_class = ""
+++

