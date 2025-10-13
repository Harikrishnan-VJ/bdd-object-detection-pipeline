import json
from collections import Counter, defaultdict
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_data_labels_path = '/home/user/hari/test/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json'
val_data_labels_path = '/home/user/hari/test/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json'


def load_json_data(file_path):
    """Load JSON data from a file.

    Args:
        file_path (str): Path to the JSON file.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file path {file_path} does not exist.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file at {file_path} is not a valid JSON.")
        return None
    
# Detection classes to focus on (from BDD100K docs)
DETECTION_CLASSES = {
    "bike", "bus", "car", "motor", "person", "rider",
    "traffic light", "traffic sign", "train", "truck"
}


def filter_detection_labels(images):
    """
    Filter labels to only include detection classes with bounding boxes.

    Args:
        images (list): List of image dicts.

    Returns:
        list: Filtered images with only detection labels.
    """
    filtered = []
    empty_images = []
    
    for img in images:
        # Filter labels. Keep only classes with bboxes
        det_labels = [label for label in img.get('labels', [])
                      if 'box2d' in label]
        
        if det_labels:  # Keep even if no labels for anomaly check
            img_copy = img.copy()
            img_copy['labels'] = det_labels
            filtered.append(img_copy)
        else:
            print("empty image found")
            empty_images.append(img['name'])

    return filtered, empty_images

def compute_basic_stats(images, split_name, empty_images):
    """
    Compute basic stats: images, bboxes, classes.

    Args:
        images (list): Filtered image list.
        split_name (str): 'train' or 'val'.

    Returns:
        dict: Stats dictionary.
    """
    num_images = len(images)
    class_counts = Counter()
    bboxes_per_image = []
    attr_weather = Counter()
    attr_scene = Counter()
    attr_timeofday = Counter()
    traffic_light_colors = Counter()  
    occluded_counts = Counter()
    truncated_counts = Counter()

    for img in images:
        num_bboxes = len(img['labels'])
        bboxes_per_image.append(num_bboxes)
        attr_weather[img['attributes'].get('weather', 'unknown')] += 1
        attr_scene[img['attributes'].get('scene', 'unknown')] += 1
        attr_timeofday[img['attributes'].get('timeofday', 'unknown')] += 1

        for label in img['labels']:
            cat = label['category']
            class_counts[cat] += 1
            attrs = label.get('attributes', {})
            if cat == 'traffic light':
                traffic_light_colors[attrs.get('trafficLightColor', 'none')] += 1
            occluded_counts[(cat, attrs.get('occluded', False))] += 1
            truncated_counts[(cat, attrs.get('truncated', False))] += 1

    unique_classes = list(class_counts.keys())
    total_bboxes = sum(bboxes_per_image)
    avg_bboxes = np.mean(bboxes_per_image) if bboxes_per_image else 0
    min_bboxes = min(bboxes_per_image) if bboxes_per_image else 0
    max_bboxes = max(bboxes_per_image) if bboxes_per_image else 0
    no_bbox_images = sum(1 for n in bboxes_per_image if n == 0)

    # Box sizes (width, height, area)
    box_sizes = defaultdict(list)
    for img in images:
        for label in img['labels']:
            box = label['box2d']
            w = box['x2'] - box['x1']
            h = box['y2'] - box['y1']
            area = w * h
            box_sizes[label['category']].append((w, h, area))

    avg_sizes = {cat: (np.mean([s[0] for s in sizes]), np.mean([s[1] for s in sizes]), np.mean([s[2] for s in sizes]))
                 for cat, sizes in box_sizes.items()}

    # Anomalies: images with >50 bboxes as example
    anomaly_high_bboxes = [img['name'] for img, n in zip(images, bboxes_per_image) if n > 50]

    # Patterns: class co-occurrence (simple count of pairs)
    co_occurs = defaultdict(int)
    for img in images:
        cats = set(label['category'] for label in img['labels'])
        for c1 in cats:
            for c2 in cats:
                if c1 < c2:
                    co_occurs[(c1, c2)] += 1
                    

    stats = {
        'split': split_name,
        'num_images': num_images,
        'empty_images': empty_images,
        'total_bboxes': total_bboxes,
        'unique_classes': unique_classes,
        'num_unique_classes': len(unique_classes),
        'class_counts': dict(class_counts),
        'avg_bboxes_per_image': avg_bboxes,
        'min_bboxes_per_image': min_bboxes,
        'max_bboxes_per_image': max_bboxes,
        'no_bbox_images': no_bbox_images,
        'weather_dist': dict(attr_weather),
        'scene_dist': dict(attr_scene),
        'timeofday_dist': dict(attr_timeofday),
        'traffic_light_colors': dict(traffic_light_colors),
        'occluded_per_class': {f"{cat}_{occ}": cnt for (cat, occ), cnt in occluded_counts.items()},
        'truncated_per_class': {f"{cat}_{trunc}": cnt for (cat, trunc), cnt in truncated_counts.items()},
        'avg_box_sizes': {cat: {'avg_width': aw, 'avg_height': ah, 'avg_area': aa} for cat, (aw, ah, aa) in avg_sizes.items()},
        'anomaly_high_bboxes_images': anomaly_high_bboxes,
        'class_co_occurrences': {f"{c1}_{c2}": cnt for (c1, c2), cnt in co_occurs.items() if cnt > 0}
    }
    return stats

def generate_plots(stats, output_dir):
    """
    Generate visualization plots and save to output_dir.

    Args:
        stats (dict): Stats from compute_basic_stats.
        output_dir (str): Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    split = stats['split']

    # Bar chart: class counts
    df_classes = pd.DataFrame(list(stats['class_counts'].items()), columns=['Class', 'Count'])
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Class', y='Count', data=df_classes)
    plt.title(f'Class Distribution - {split}')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, f'class_dist_{split}.png'))
    plt.close()

    # Pie chart: weather
    plt.figure(figsize=(8, 8))
    plt.pie(stats['weather_dist'].values(), labels=stats['weather_dist'].keys(), autopct='%1.1f%%')
    plt.title(f'Weather Distribution - {split}')
    plt.savefig(os.path.join(output_dir, f'weather_dist_{split}.png'))
    plt.close()
    
    # Pie chart: empty images proportion
    total_images = stats['num_images'] + len(stats['empty_images'])
    empty_count = len(stats['empty_images'])
    non_empty_count = stats['num_images']
    plt.figure(figsize=(8, 8))
    plt.pie([empty_count, non_empty_count], labels=['Empty Images', 'Non-Empty Images'], autopct='%1.1f%%')
    plt.title(f'Empty vs Non-Empty Images - {split}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'empty_images_{split}.png'))
    plt.close()
    
    # Pie chart: scene
    plt.figure(figsize=(8, 8))
    plt.pie(stats['scene_dist'].values(), labels=stats['scene_dist'].keys(), autopct='%1.1f%%')
    plt.title(f'Scene Distribution - {split}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'scene_dist_{split}.png'))
    plt.close()


    

def write_report(train_stats, val_stats, output_file):
    """
    Write analysis report to Markdown file.

    Args:
        train_stats (dict): Train stats.
        val_stats (dict): Val stats.
        output_file (str): Path to output MD file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# BDD100K Object Detection Analysis Report\n\n")

        f.write("## Basic Overview\n")
        f.write(f"- Total images: Train={train_stats['num_images']}, Val={val_stats['num_images']}\n")
        f.write(f"- Images with no labels: Train={len(train_stats['empty_images'])}, Val={len(val_stats['empty_images'])}\n")
        f.write(f"- Total bboxes: Train={train_stats['total_bboxes']}, Val={val_stats['total_bboxes']}\n")
        f.write(f"- Unique classes: {', '.join(set(train_stats['unique_classes'] + val_stats['unique_classes']))}\n")
        f.write(f"- Num unique classes: {len(set(train_stats['unique_classes'] + val_stats['unique_classes']))}\n\n")

        f.write("## Class Distribution\n")
        f.write("Train:\n")
        for cls, cnt in train_stats['class_counts'].items():
            f.write(f"- {cls}: {cnt}\n")
        f.write("Val:\n")
        for cls, cnt in val_stats['class_counts'].items():
            f.write(f"- {cls}: {cnt}\n\n")

        f.write("## Bounding Box Stats\n")
        f.write(f"- Avg bboxes per image: Train={train_stats['avg_bboxes_per_image']:.2f}, Val={val_stats['avg_bboxes_per_image']:.2f}\n")
        f.write(f"- Min/Max bboxes per image: Train={train_stats['min_bboxes_per_image']}/{train_stats['max_bboxes_per_image']}, Val={val_stats['min_bboxes_per_image']}/{val_stats['max_bboxes_per_image']}\n")
        f.write(f"- Images with no bboxes: Train={train_stats['no_bbox_images']}, Val={val_stats['no_bbox_images']}\n\n")

        f.write("## Attribute Distributions\n")
        f.write("Weather (Train/Val):\n")
        all_weathers = set(train_stats['weather_dist'].keys()) | set(val_stats['weather_dist'].keys())
        for w in all_weathers:
            f.write(f"- {w}: {train_stats['weather_dist'].get(w, 0)} / {val_stats['weather_dist'].get(w, 0)}\n")
        # Add similar for scene, timeofday

        f.write("\n## Traffic Light Colors (Train/Val)\n")
        all_colors = set(train_stats['traffic_light_colors'].keys()) | set(val_stats['traffic_light_colors'].keys())
        for c in all_colors:
            f.write(f"- {c}: {train_stats['traffic_light_colors'].get(c, 0)} / {val_stats['traffic_light_colors'].get(c, 0)}\n")

        f.write("\n## Occluded and Truncated Stats\n")
        # Summarize similarly

        f.write("\n## Average Box Sizes per Class (Width/Height/Area)\n")
        all_classes = set(train_stats['avg_box_sizes'].keys()) | set(val_stats['avg_box_sizes'].keys())
        for cls in all_classes:
            t_sizes = train_stats['avg_box_sizes'].get(cls, {'avg_width': 0, 'avg_height': 0, 'avg_area': 0})
            v_sizes = val_stats['avg_box_sizes'].get(cls, {'avg_width': 0, 'avg_height': 0, 'avg_area': 0})
            f.write(f"- {cls}: Train={t_sizes['avg_width']:.2f}/{t_sizes['avg_height']:.2f}/{t_sizes['avg_area']:.2f}, Val={v_sizes['avg_width']:.2f}/{v_sizes['avg_height']:.2f}/{v_sizes['avg_area']:.2f}\n")

        f.write("\n## Anomalies\n")
        f.write(f"- Images with >50 bboxes: Train={', '.join(train_stats['anomaly_high_bboxes_images'])}, Val={', '.join(val_stats['anomaly_high_bboxes_images'])}\n")

        f.write("\n## Patterns: Class Co-occurrences (Train)\n")
        for pair, cnt in sorted(train_stats['class_co_occurrences'].items(), key=lambda x: x[1], reverse=True)[:10]:  # Top 10
            f.write(f"- {pair}: {cnt}\n")
        # Add for val

        f.write("\n## Visualizations\n")
        f.write("See plots in output/plots/ for charts (e.g., class_dist_train.png).\n")
        # Add more details on patterns/anomalies as needed

def main():
    """Main function to run analysis."""
    train_path = '/home/user/hari/test/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json'
    val_path = '/home/user/hari/test/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json'
    output_dir = 'output'
    plots_dir = os.path.join(output_dir, 'plots')
    report_file = os.path.join(output_dir, 'analysis_report.md')

    os.makedirs(output_dir, exist_ok=True)

    train_images, empty_train_images = filter_detection_labels(load_json_data(train_path))
    val_images, empty_val_images = filter_detection_labels(load_json_data(val_path))

    train_stats = compute_basic_stats(train_images, 'train', empty_train_images)
    val_stats = compute_basic_stats(val_images, 'val', empty_val_images)

    generate_plots(train_stats, plots_dir)
    generate_plots(val_stats, plots_dir)

    write_report(train_stats, val_stats, report_file)
    print(f"Analysis complete. Report saved to {report_file}")

if __name__ == "__main__":
    main()