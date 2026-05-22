
# Spinal cord neuron stimulation simulation

# How to run this code:
# 1. Put this file in the same folder as these two files:
#    - afbeelding ruggenmerg makaak .png
#    - metaMN.csv
# 2. Install the packages if Python does not know them yet:
#    pip install opencv-python numpy pandas matplotlib
# 3. Run the code in the terminal with:
#    python final_neuron_stimulation_beginner_comments.py
# 4. A window opens with the image, neurons and sliders.
# 5. Move the sliders to change the electrode settings.
# 6. Type a muscle ID in the target muscle box.
# 7. Click Fast optimize to search for the best electrode setting.
# 8. When the plot closes, the program saves CSV files with the results.


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button


# These are the main settings I can change before I run the code

image_file = "afbeelding ruggenmerg makaak .png"  # this is the spinal cord image
csv_file = "metaMN.csv"  # this file contains the muscle data

number_of_neurons = 500  # this is how many neurons I simulate

muscle_x_shift = 46.1  # moves the muscle points left or right
muscle_y_shift = 33.8  # moves the muscle points up or down
muscle_scale = 7.4  # makes the muscle point map bigger or smaller

max_muscle_distance = 2.0  # maximum distance for a neuron to be linked to a muscle

mu0 = 4 * np.pi * 10**-7
meters_per_unit = 0.0001

target_muscle = 1  # this is the starting target muscle

off_target_penalty = 3.0
current_penalty = 0.5

# These are the values the program will try during optimization
# I use bigger steps because small steps made the program too slow on my computer
optimization_x_values = np.arange(30, 71, 3)
optimization_y_values = np.arange(15, 56, 3)
optimization_current_values = np.arange(0.02, 0.51, 0.03)

best_result = None
all_results_dataframe = None


# Load the spinal cord image
# The image file must be in the same folder as this Python file

image = cv2.imread(image_file)

if image is None:
    print("Image not found. Check the image file name.")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

height, width, channels = image.shape

print("Image width:", width)
print("Image height:", height)


# Make a mask for the blue and/or purple spinal cord area
# This mask decides where the simulated neurons are allowed to be placed

lower_blue = np.array([70, 3, 80])
upper_blue = np.array([170, 80, 255])

blue_mask = cv2.inRange(image_hsv, lower_blue, upper_blue)  # pixels inside this color range become white in the mask

kernel = np.ones((3, 3), np.uint8)  # small square used to clean the mask

blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)


# Put the simulated neurons inside the selected blue/purple mask

y_pixels, x_pixels = np.where(blue_mask > 0)

top_cutoff = np.percentile(y_pixels, 8)
keep_pixels = y_pixels > top_cutoff

y_pixels = y_pixels[keep_pixels]
x_pixels = x_pixels[keep_pixels]

if len(x_pixels) < number_of_neurons:
    print("Not enough blue pixels.")
    exit()

np.random.seed(1)  # this makes the random neuron positions the same every time I run the code

random_indices = np.random.choice(
    len(x_pixels),
    number_of_neurons,
    replace=False
)

neuron_x_pixels = x_pixels[random_indices]
neuron_y_pixels = y_pixels[random_indices]

simulated_neurons = pd.DataFrame()  # this table stores all neuron information

simulated_neurons["neuron_id"] = range(1, number_of_neurons + 1)

simulated_neurons["x"] = neuron_x_pixels / width * 100
simulated_neurons["y"] = 100 - (neuron_y_pixels / height * 100)

simulated_neurons["field_strength"] = 0.0
simulated_neurons["threshold"] = 20.0
simulated_neurons["fires"] = False
simulated_neurons["nearest_muscle"] = -1
simulated_neurons["nearest_muscle_distance"] = 999.0
simulated_neurons["activates_muscle"] = False


# Load the muscle data from the CSV file

muscle_data = pd.read_csv(csv_file)

if "Unnamed: 6" in muscle_data.columns:
    muscle_data = muscle_data.drop(columns=["Unnamed: 6"])

muscle_data = muscle_data.dropna()

segment_id = 8  # I only use muscle points from this segment

muscle_points = muscle_data[muscle_data["idSegment"] == segment_id].copy()

print("Number of muscle points in segment 8:", len(muscle_points))

muscle_points["x"] = muscle_points["nX"]
muscle_points["y"] = muscle_points["nY"]

muscle_points["x_centered"] = muscle_points["x"] - muscle_points["x"].mean()
muscle_points["y_centered"] = muscle_points["y"] - muscle_points["y"].mean()

max_abs_x = abs(muscle_points["x_centered"]).max()
max_abs_y = abs(muscle_points["y_centered"]).max()

common_max = max(max_abs_x, max_abs_y)  # use one scale so x and y stay in proportion

muscle_points["x_normalized"] = muscle_points["x_centered"] / common_max
muscle_points["y_normalized"] = muscle_points["y_centered"] / common_max

muscle_points["x_plot"] = (
    muscle_x_shift + muscle_points["x_normalized"] * muscle_scale
)

muscle_points["y_plot"] = (
    muscle_y_shift - muscle_points["y_normalized"] * muscle_scale
)


# I also store some columns as NumPy arrays
# This makes the calculations faster when the optimization is running

neuron_x_values = simulated_neurons["x"].values
neuron_y_values = simulated_neurons["y"].values

muscle_x_values = muscle_points["x_plot"].values
muscle_y_values = muscle_points["y_plot"].values
muscle_id_values = muscle_points["idMuscle"].values


# This function calculates the field strength around the electrode

def calculate_field_strength(current, distance_units):
    distance_meters = distance_units * meters_per_unit
    distance_meters = distance_meters + 0.000001

    field_tesla = (mu0 * current) / (2 * np.pi * distance_meters)
    field_microtesla = field_tesla * 1000000

    return field_microtesla


# This function finds the closest muscle point for every neuron

def assign_nearest_muscle_to_neurons():
    nearest_muscles = []
    nearest_distances = []

    for i in range(len(simulated_neurons)):
        neuron_x = simulated_neurons.loc[i, "x"]
        neuron_y = simulated_neurons.loc[i, "y"]

        distances = np.sqrt(
            (muscle_x_values - neuron_x) ** 2 +
            (muscle_y_values - neuron_y) ** 2
        )

        nearest_index = np.argmin(distances)  # index of the closest muscle point

        nearest_muscle = muscle_id_values[nearest_index]
        nearest_distance = distances[nearest_index]

        nearest_muscles.append(nearest_muscle)
        nearest_distances.append(nearest_distance)

    simulated_neurons["nearest_muscle"] = nearest_muscles
    simulated_neurons["nearest_muscle_distance"] = nearest_distances


assign_nearest_muscle_to_neurons()

nearest_muscle_array = simulated_neurons["nearest_muscle"].values
nearest_distance_array = simulated_neurons["nearest_muscle_distance"].values


# This function updates which neurons are firing after the sliders change

def update_neurons(electrode_x, electrode_y, current, threshold):
    distance = np.sqrt(
        (simulated_neurons["x"] - electrode_x) ** 2 +
        (simulated_neurons["y"] - electrode_y) ** 2
    )

    simulated_neurons["field_strength"] = calculate_field_strength(
        current,
        distance
    )

    simulated_neurons["threshold"] = threshold

    simulated_neurons["fires"] = (
        simulated_neurons["field_strength"] >= simulated_neurons["threshold"]
    )

    simulated_neurons["activates_muscle"] = (
        (simulated_neurons["fires"] == True) &
        (simulated_neurons["nearest_muscle_distance"] <= max_muscle_distance)
    )


# This function makes a short text summary for the plot

def make_muscle_summary():
    active_neurons = simulated_neurons[
        simulated_neurons["activates_muscle"] == True
    ]

    if len(active_neurons) == 0:
        return "Activated muscles:\nNo muscle activation"

    counts = active_neurons["nearest_muscle"].value_counts()
    counts = counts.sort_values(ascending=False)

    text = "Activated muscles:\n"

    for muscle_id, count in counts.head(8).items():
        text = (
            text
            + "idMuscle "
            + str(int(muscle_id))
            + ": "
            + str(count)
            + "\n"
        )

    return text


def make_offsets(dataframe):
    if len(dataframe) == 0:
        return np.empty((0, 2))

    return np.column_stack((dataframe["x"], dataframe["y"]))


# This function test one possible stimulation setting

def test_stimulation_setting(electrode_x, electrode_y, current, threshold, target):
    distance = np.sqrt(
        (neuron_x_values - electrode_x) ** 2 +
        (neuron_y_values - electrode_y) ** 2
    )

    field_strength = calculate_field_strength(current, distance)

    fires = field_strength >= threshold

    muscle_linked = fires & (nearest_distance_array <= max_muscle_distance)

    target_active = muscle_linked & (nearest_muscle_array == target)
    other_active = muscle_linked & (nearest_muscle_array != target)

    target_count = np.sum(target_active)
    other_count = np.sum(other_active)
    total_firing = np.sum(fires)
    total_muscle_linked = np.sum(muscle_linked)

    selectivity_ratio = target_count / (other_count + 1)  # +1 prevents division by zero and reuslting in an error

    score = (
        target_count
        - off_target_penalty * other_count
        - current_penalty * current
    )

    result = {
        "electrode_x": electrode_x,
        "electrode_y": electrode_y,
        "current": current,
        "threshold": threshold,
        "target_muscle": target,
        "target_muscle_count": int(target_count),
        "other_muscle_count": int(other_count),
        "total_firing_neurons": int(total_firing),
        "total_muscle_linked_neurons": int(total_muscle_linked),
        "selectivity_ratio_target_div_other": selectivity_ratio,
        "optimization_score": score
    }

    return result


# This function tries many electrode positions and current values
# It then chooses the best result for the target muscle

def optimize_target_muscle(target):
    global all_results_dataframe

    threshold = slider_threshold.val

    results = []

    print("")
    print("Starting fast optimization...")
    print("Target muscle:", target)

    total_tests = (
        len(optimization_x_values)
        * len(optimization_y_values)
        * len(optimization_current_values)
    )

    counter = 0

    for electrode_x in optimization_x_values:
        for electrode_y in optimization_y_values:
            for current in optimization_current_values:
                result = test_stimulation_setting(
                    electrode_x,
                    electrode_y,
                    current,
                    threshold,
                    target
                )

                results.append(result)
                counter = counter + 1

                if counter % 1000 == 0:
                    print("Progress:", counter, "/", total_tests)

    results_dataframe = pd.DataFrame(results)
    all_results_dataframe = results_dataframe

    results_dataframe.to_csv(
        "optimization_all_results_fast.csv",
        index=False
    )

    useful_results = results_dataframe[
        results_dataframe["target_muscle_count"] > 0
    ].copy()

    if len(useful_results) == 0:
        print("No useful result found.")
        return None

    best_results = useful_results.sort_values(
        by=[
            "optimization_score",
            "selectivity_ratio_target_div_other",
            "target_muscle_count",
            "current"
        ],
        ascending=[
            False,
            False,
            False,
            True
        ]
    )

    best_results.to_csv(
        "optimization_best_results_fast.csv",
        index=False
    )

    best = best_results.iloc[0]  # first row is the best result after the sorting

    print("")
    print("BEST OPTIMUM")
    print("============")
    print("Target muscle:", int(best["target_muscle"]))
    print("Best x:", best["electrode_x"])
    print("Best y:", best["electrode_y"])
    print("Best current:", best["current"])
    print("Target count:", int(best["target_muscle_count"]))
    print("Other count:", int(best["other_muscle_count"]))
    print("Ratio:", best["selectivity_ratio_target_div_other"])
    print("Score:", best["optimization_score"])

    return best


# Show the best result in a separate small window
# I did this because too much text inside the main image became unclear

def show_result_popup(best):
    if best is None:
        return

    popup_text = (
        "BEST OPTIMUM\n\n"
        + "Target muscle: idMuscle "
        + str(int(best["target_muscle"]))
        + "\n\n"
        + "Best electrode x: "
        + str(round(float(best["electrode_x"]), 2))
        + "\n"
        + "Best electrode y: "
        + str(round(float(best["electrode_y"]), 2))
        + "\n"
        + "Best current: "
        + str(round(float(best["current"]), 3))
        + "\n"
        + "Threshold: "
        + str(round(float(best["threshold"]), 2))
        + "\n\n"
        + "Target muscle neurons: "
        + str(int(best["target_muscle_count"]))
        + "\n"
        + "Other muscle neurons: "
        + str(int(best["other_muscle_count"]))
        + "\n"
        + "Selectivity ratio: "
        + str(round(float(best["selectivity_ratio_target_div_other"]), 3))
        + "\n"
        + "Optimization score: "
        + str(round(float(best["optimization_score"]), 3))
    )

    popup_figure, popup_axis = plt.subplots(figsize=(5, 4))
    popup_axis.axis("off")

    popup_axis.text(
        0.05,
        0.95,
        popup_text,
        fontsize=11,
        verticalalignment="top"
    )

    plt.show()


# These are the starting values when the plot first opens

start_electrode_x = 50
start_electrode_y = 35
start_current = 0.05
start_threshold = 20.0

update_neurons(
    start_electrode_x,
    start_electrode_y,
    start_current,
    start_threshold
)


# Make the main plot with the spinal cord image, neurons and muscle points

figure, axis = plt.subplots(figsize=(10, 8))

plt.subplots_adjust(bottom=0.42, right=0.72)

axis.imshow(
    image_rgb,
    extent=[0, 100, 0, 100],
    origin="upper",
    alpha=0.75
)

axis.scatter(
    muscle_points["x_plot"],
    muscle_points["y_plot"],
    s=8,
    color="orange",
    alpha=0.4,
    label="muscle map points"
)

not_firing_neurons = simulated_neurons[simulated_neurons["fires"] == False]
firing_neurons = simulated_neurons[simulated_neurons["fires"] == True]

not_firing_plot = axis.scatter(
    not_firing_neurons["x"],
    not_firing_neurons["y"],
    s=5,
    color="gray",
    alpha=0.5,
    label="not firing"
)

firing_plot = axis.scatter(
    firing_neurons["x"],
    firing_neurons["y"],
    s=12,
    color="blue",
    label="firing"
)

target_plot = axis.scatter(
    [],
    [],
    s=35,
    color="red",
    label="target muscle firing"
)

best_marker = axis.scatter(
    [],
    [],
    s=250,
    color="red",
    marker="*",
    label="best electrode"
)

electrode_circle = plt.Circle(
    (start_electrode_x, start_electrode_y),
    1.5,
    fill=False,
    color="black",
    linewidth=2
)

axis.add_patch(electrode_circle)

field_circle_1 = plt.Circle(
    (start_electrode_x, start_electrode_y),
    5,
    fill=False,
    color="black",
    linestyle="--",
    alpha=0.4
)

field_circle_2 = plt.Circle(
    (start_electrode_x, start_electrode_y),
    10,
    fill=False,
    color="black",
    linestyle="--",
    alpha=0.25
)

field_circle_3 = plt.Circle(
    (start_electrode_x, start_electrode_y),
    15,
    fill=False,
    color="black",
    linestyle="--",
    alpha=0.15
)

axis.add_patch(field_circle_1)
axis.add_patch(field_circle_2)
axis.add_patch(field_circle_3)

summary_text = figure.text(
    0.74,
    0.82,
    make_muscle_summary(),
    fontsize=9,
    verticalalignment="top",
    horizontalalignment="left",
    bbox=dict(facecolor="white", alpha=0.85)
)

optimization_text = figure.text(
    0.74,
    0.55,
    "Optimization result will appear here",
    fontsize=8,
    verticalalignment="top",
    horizontalalignment="left",
    bbox=dict(facecolor="white", alpha=0.85)
)

axis.set_xlim(0, 100)
axis.set_ylim(0, 100)
axis.set_xlabel("x position")
axis.set_ylabel("y position")
axis.set_aspect("equal")
axis.legend(loc="lower right", fontsize=8)


# Make the sliders, text box and button for the user

slider_x_axis = plt.axes([0.20, 0.33, 0.60, 0.03])
slider_y_axis = plt.axes([0.20, 0.28, 0.60, 0.03])
slider_current_axis = plt.axes([0.20, 0.23, 0.60, 0.03])
slider_threshold_axis = plt.axes([0.20, 0.18, 0.60, 0.03])
slider_distance_axis = plt.axes([0.20, 0.13, 0.60, 0.03])

target_box_axis = plt.axes([0.20, 0.07, 0.20, 0.04])
button_axis = plt.axes([0.45, 0.07, 0.35, 0.04])

slider_x = Slider(
    slider_x_axis,
    "electrode x",
    0,
    100,
    valinit=start_electrode_x
)

slider_y = Slider(
    slider_y_axis,
    "electrode y",
    0,
    100,
    valinit=start_electrode_y
)

slider_current = Slider(
    slider_current_axis,
    "current",
    0.0,
    0.5,
    valinit=start_current
)

slider_threshold = Slider(
    slider_threshold_axis,
    "threshold",
    1.0,
    100.0,
    valinit=start_threshold
)

slider_distance = Slider(
    slider_distance_axis,
    "muscle distance",
    0.1,
    8.0,
    valinit=max_muscle_distance
)

target_box = TextBox(
    target_box_axis,
    "target muscle",
    initial=str(target_muscle)
)

optimize_button = Button(
    button_axis,
    "Fast optimize"
)


# This function runs every time the user moves a slider

def update_plot(value):
    global max_muscle_distance

    electrode_x = slider_x.val
    electrode_y = slider_y.val
    current = slider_current.val
    threshold = slider_threshold.val

    max_muscle_distance = slider_distance.val

    update_neurons(electrode_x, electrode_y, current, threshold)

    not_firing_neurons = simulated_neurons[simulated_neurons["fires"] == False]
    firing_neurons = simulated_neurons[simulated_neurons["fires"] == True]

    not_firing_plot.set_offsets(make_offsets(not_firing_neurons))
    firing_plot.set_offsets(make_offsets(firing_neurons))

    target_active_neurons = simulated_neurons[
        (simulated_neurons["fires"] == True) &
        (simulated_neurons["nearest_muscle"] == target_muscle) &
        (simulated_neurons["nearest_muscle_distance"] <= max_muscle_distance)
    ]

    target_plot.set_offsets(make_offsets(target_active_neurons))

    electrode_circle.center = (electrode_x, electrode_y)
    field_circle_1.center = (electrode_x, electrode_y)
    field_circle_2.center = (electrode_x, electrode_y)
    field_circle_3.center = (electrode_x, electrode_y)

    summary_text.set_text(make_muscle_summary())

    number_firing = len(firing_neurons)

    number_muscle_active = len(
        simulated_neurons[simulated_neurons["activates_muscle"] == True]
    )

    axis.set_title(
        "Firing neurons: "
        + str(number_firing)
        + " | muscle-linked neurons: "
        + str(number_muscle_active)
        + " | target muscle: "
        + str(target_muscle)
    )

    figure.canvas.draw_idle()  # redraw the plot after changing the values


# This function runs when the user clicks the Fast optimize button

def optimize_button_clicked(event):
    global target_muscle
    global best_result

    try:
        target_muscle = int(target_box.text)
    except:
        optimization_text.set_text(
            "Please enter a valid muscle ID, for example 1 or 3"
        )

        figure.canvas.draw_idle()  # redraw the plot after changing values
        return

    best = optimize_target_muscle(target_muscle)

    if best is None:
        optimization_text.set_text(
            "No optimum found for idMuscle " + str(target_muscle)
        )

        figure.canvas.draw_idle()  # redraw the plot after changing values
        return

    best_result = best

    best_x = float(best["electrode_x"])
    best_y = float(best["electrode_y"])
    best_current = float(best["current"])

    slider_x.set_val(best_x)
    slider_y.set_val(best_y)
    slider_current.set_val(best_current)

    best_marker.set_offsets([[best_x, best_y]])

    text = (
        "OPTIMUM for idMuscle "
        + str(target_muscle)
        + "\n"
        + "x = "
        + str(round(best_x, 2))
        + ", y = "
        + str(round(best_y, 2))
        + ", current = "
        + str(round(best_current, 3))
        + "\n"
        + "target count = "
        + str(int(best["target_muscle_count"]))
        + ", other count = "
        + str(int(best["other_muscle_count"]))
        + "\n"
        + "ratio = "
        + str(round(float(best["selectivity_ratio_target_div_other"]), 3))
        + "\n"
        + "score = "
        + str(round(float(best["optimization_score"]), 3))
    )

    optimization_text.set_text(text)

    update_plot(None)

    show_result_popup(best)


# Connect the sliders and button to the functions above

slider_x.on_changed(update_plot)
slider_y.on_changed(update_plot)
slider_current.on_changed(update_plot)
slider_threshold.on_changed(update_plot)
slider_distance.on_changed(update_plot)

optimize_button.on_clicked(optimize_button_clicked)

update_plot(None)

plt.show()


# Save the final neuron data after the plot is closed

simulated_neurons.to_csv(
    "neuron_muscle_activation_result.csv",
    index=False
)

print("Saved file: neuron_muscle_activation_result.csv")
print("Saved file: optimization_all_results_fast.csv")
print("Saved file: optimization_best_results_fast.csv")
print(make_muscle_summary())