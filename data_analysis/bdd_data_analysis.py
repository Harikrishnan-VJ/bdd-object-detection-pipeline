import json
from collections import Counter, defaultdict
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



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
    Filter labels with only bounding boxes.

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
        
        if det_labels:  
            img_copy = img.copy()
            img_copy['labels'] = det_labels
            filtered.append(img_copy)
        else:
            print("Empty image found")
            empty_images.append(img['name'])

    return filtered, empty_images


def compute_basic_stats(images, split_name, empty_images, total_image_count):
    """
    Compute basic stats: images, bboxes, classes, and anomalies.

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
    anomaly_high_bboxes = []  
    anomaly_small_bboxes = []  
    anomaly_large_bboxes = []  
    anomaly_high_class_count = []  

    for img in images:
        num_bboxes = len(img['labels'])
        bboxes_per_image.append(num_bboxes)
        attr_weather[img['attributes'].get('weather', 'unknown')] += 1
        attr_scene[img['attributes'].get('scene', 'unknown')] += 1
        attr_timeofday[img['attributes'].get('timeofday', 'unknown')] += 1

        # Per-image class counts for high class count anomaly
        img_class_counts = Counter(label['category'] for label in img['labels'])

        for label in img['labels']:
            cat = label['category']
            class_counts[cat] += 1
            attrs = label.get('attributes', {})
            if cat == 'traffic light':
                traffic_light_colors[attrs.get('trafficLightColor', 'none')] += 1
            occluded_counts[(cat, attrs.get('occluded', False))] += 1
            truncated_counts[(cat, attrs.get('truncated', False))] += 1

            # Compute box size for anomaly detection
            box = label['box2d']
            w = box['x2'] - box['x1']
            h = box['y2'] - box['y1']
            area = w * h
            # Check for small or large bounding box anomalies
            if area < 20:
                anomaly_small_bboxes.append((img['name'], cat, area))
            if area > 500000:
                anomaly_large_bboxes.append((img['name'], cat, area))

        # Check for high class count anomaly
        for cls, count in img_class_counts.items():
            if count > 30:
                anomaly_high_class_count.append((img['name'], cls, count))

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

    # Anomaly images with >50 bboxes
    anomaly_high_bboxes = [img['name'] for img, n in zip(images, bboxes_per_image) if n > 50]

    # Patterns - class co-occurrence
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
        'background_images': total_image_count - num_images - len(empty_images),
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
        'anomaly_small_bboxes': anomaly_small_bboxes,  
        'anomaly_large_bboxes': anomaly_large_bboxes,  
        'anomaly_high_class_count': anomaly_high_class_count,  
        'class_co_occurrences': {f"{c1}_{c2}": cnt for (c1, c2), cnt in co_occurs.items() if cnt > 0}
    }
    return stats


def generate_plots(stats, output_dir):
    """
    Generate visualization plots and save to the output_dir.

    Args:
        stats (dict): Stats from compute_basic_stats.
        output_dir (str): Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    split = stats['split']

    # Bar chart for class counts
    df_classes = pd.DataFrame(list(stats['class_counts'].items()), columns=['Class', 'Count'])
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Class', y='Count', data=df_classes)
    plt.title(f'Class Distribution - {split}')
    plt.xticks(rotation=45)
    os.makedirs(os.path.join(output_dir, 'class_distribution'), exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'class_distribution', f'class_dist_{split}.png'))
    plt.close()

    # Pie chart for weather
    plt.figure(figsize=(8, 8))
    plt.pie(stats['weather_dist'].values(), labels=stats['weather_dist'].keys(), autopct='%1.1f%%')
    plt.title(f'Weather Distribution - {split}')
    os.makedirs(os.path.join(output_dir, 'weather_distribution'), exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'weather_distribution', f'weather_dist_{split}.png'))
    plt.close()
    
    # Pie chart for background images proportion
    total_images = stats['num_images'] + len(stats['empty_images'])
    background_img_count = stats['background_images']
    obj_img_count = stats['num_images']
    plt.figure(figsize=(8, 8))
    plt.pie([background_img_count, obj_img_count], labels=['background_images', 'Object Images'], autopct='%1.1f%%')
    plt.title(f'Background vs Object Images - {split}')
    plt.tight_layout()
    os.makedirs(os.path.join(output_dir, 'background_images'), exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'background_images', f'background_images_{split}.png'))
    plt.close()
    
    
    # Pie chart for empty images proportion
    total_images = stats['num_images'] + len(stats['empty_images'])
    empty_count = len(stats['empty_images'])
    non_empty_count = stats['num_images']
    plt.figure(figsize=(8, 8))
    plt.pie([empty_count, non_empty_count], labels=['Empty Images', 'Non-Empty Images'], autopct='%1.1f%%')
    plt.title(f'Empty vs Non-Empty Images - {split}')
    plt.tight_layout()
    os.makedirs(os.path.join(output_dir, 'empty_images'), exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'empty_images', f'empty_images_{split}.png'))
    plt.close()
    
    # Pie chart for scene distribution
    plt.figure(figsize=(8, 8))
    plt.pie(stats['scene_dist'].values(), labels=stats['scene_dist'].keys(), autopct='%1.1f%%')
    plt.title(f'Scene Distribution - {split}')
    plt.tight_layout()
    os.makedirs(os.path.join(output_dir, 'scene_distibution'), exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'scene_distibution', f'scene_dist_{split}.png'))
    plt.close()
    
    # Pie chart for time of day distribution
    plt.figure(figsize=(8, 8))
    plt.pie(stats['timeofday_dist'].values(), labels=stats['timeofday_dist'].keys(), autopct='%1.1f%%')
    plt.title(f'Time of Day Distribution - {split}')
    plt.tight_layout()
    os.makedirs(os.path.join(output_dir, 'time_of_day_distribution'), exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'time_of_day_distribution', f'timeofday_dist_{split}.png'))
    plt.close()
    
    # Bar chart for occluded counts per class
    df_occluded = pd.DataFrame([(k.split('_')[0], k.split('_')[1], v) for k, v in stats['occluded_per_class'].items()],
                               columns=['Class', 'Occluded', 'Count'])
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Class', y='Count', hue='Occluded', data=df_occluded)
    plt.title(f'Occluded Counts per Class - {split}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs(os.path.join(output_dir, 'occluded_objects_per_class'), exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'occluded_objects_per_class', f'occluded_counts_{split}.png'))
    plt.close()
    
    # Bar chart for truncated counts per class
    df_truncated = pd.DataFrame([(k.split('_')[0], k.split('_')[1], v) for k, v in stats['truncated_per_class'].items()],
                                columns=['Class', 'Truncated', 'Count'])
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Class', y='Count', hue='Truncated', data=df_truncated)
    plt.title(f'Truncated Counts per Class - {split}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs(os.path.join(output_dir, 'truncated_objects_per_class'), exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'truncated_objects_per_class', f'truncated_counts_{split}.png'))
    plt.close()


def write_report(train_stats, val_stats, output_file):
    """
    Write the analysis report to .md file

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
            f.write(f"- {cls}: {cnt}\n")
        f.write("\n")

        f.write("## Bounding Box Stats\n")
        f.write(f"- Avg bboxes per image: Train={train_stats['avg_bboxes_per_image']:.2f}, Val={val_stats['avg_bboxes_per_image']:.2f}\n")
        f.write(f"- Min/Max bboxes per image: Train={train_stats['min_bboxes_per_image']}/{train_stats['max_bboxes_per_image']}, Val={val_stats['min_bboxes_per_image']}/{val_stats['max_bboxes_per_image']}\n")
        f.write(f"- Images with no bboxes: Train={train_stats['no_bbox_images']}, Val={val_stats['no_bbox_images']}\n\n")

        f.write("## Attribute Distributions\n")
        f.write("Weather (Train/Val):\n")
        all_weathers = set(train_stats['weather_dist'].keys()) | set(val_stats['weather_dist'].keys())
        for w in sorted(all_weathers):
            f.write(f"- {w}: {train_stats['weather_dist'].get(w, 0)} / {val_stats['weather_dist'].get(w, 0)}\n")

        f.write("\nScene (Train/Val):\n")
        all_scenes = set(train_stats['scene_dist'].keys()) | set(val_stats['scene_dist'].keys())
        for s in sorted(all_scenes):
            f.write(f"- {s}: {train_stats['scene_dist'].get(s, 0)} / {val_stats['scene_dist'].get(s, 0)}\n")

        f.write("\nTime of Day (Train/Val):\n")
        all_times = set(train_stats['timeofday_dist'].keys()) | set(val_stats['timeofday_dist'].keys())
        for t in sorted(all_times):
            f.write(f"- {t}: {train_stats['timeofday_dist'].get(t, 0)} / {val_stats['timeofday_dist'].get(t, 0)}\n")

        f.write("\n## Traffic Light Colors (Train/Val)\n")
        all_colors = set(train_stats['traffic_light_colors'].keys()) | set(val_stats['traffic_light_colors'].keys())
        for c in sorted(all_colors):
            f.write(f"- {c}: {train_stats['traffic_light_colors'].get(c, 0)} / {val_stats['traffic_light_colors'].get(c, 0)}\n")

        f.write("\n## Occluded and Truncated Stats\n")
        all_classes = sorted(set(train_stats['avg_box_sizes'].keys()) | set(val_stats['avg_box_sizes'].keys()))
        f.write("Occluded per Class (Train occluded/non-occluded / Val occluded/non-occluded):\n")
        for cls in all_classes:
            t_occ = train_stats['occluded_per_class'].get(f"{cls}_True", 0)
            t_non = train_stats['occluded_per_class'].get(f"{cls}_False", 0)
            v_occ = val_stats['occluded_per_class'].get(f"{cls}_True", 0)
            v_non = val_stats['occluded_per_class'].get(f"{cls}_False", 0)
            f.write(f"- {cls}: {t_occ}/{t_non} / {v_occ}/{v_non}\n")

        f.write("\nTruncated per Class (Train truncated/non-truncated / Val truncated/non-truncated):\n")
        for cls in all_classes:
            t_trunc = train_stats['truncated_per_class'].get(f"{cls}_True", 0)
            t_non = train_stats['truncated_per_class'].get(f"{cls}_False", 0)
            v_trunc = val_stats['truncated_per_class'].get(f"{cls}_True", 0)
            v_non = val_stats['truncated_per_class'].get(f"{cls}_False", 0)
            f.write(f"- {cls}: {t_trunc}/{t_non} / {v_trunc}/{v_non}\n")

        f.write("\n## Average Box Sizes per Class (Width/Height/Area)\n")
        for cls in all_classes:
            t_sizes = train_stats['avg_box_sizes'].get(cls, {'avg_width': 0, 'avg_height': 0, 'avg_area': 0})
            v_sizes = val_stats['avg_box_sizes'].get(cls, {'avg_width': 0, 'avg_height': 0, 'avg_area': 0})
            f.write(f"- {cls}: Train={t_sizes['avg_width']:.2f}/{t_sizes['avg_height']:.2f}/{t_sizes['avg_area']:.2f}, Val={v_sizes['avg_width']:.2f}/{v_sizes['avg_height']:.2f}/{v_sizes['avg_area']:.2f}\n")

        f.write("\n## Anomalies\n")
        f.write(f"- Images with >50 bboxes: Train={', '.join(train_stats['anomaly_high_bboxes_images']) or 'None'}, Val={', '.join(val_stats['anomaly_high_bboxes_images']) or 'None'}\n\n")
        f.write(f"- Images with small bboxes (<20 pixels): Train={', '.join([f'{name} ({cat}, {area:.2f})' for name, cat, area in train_stats['anomaly_small_bboxes']]) or 'None'}, Val={', '.join([f'{name} ({cat}, {area:.2f})' for name, cat, area in val_stats['anomaly_small_bboxes']]) or 'None'}\n\n")
        f.write(f"- Images with large bboxes (>500000 pixels): Train={', '.join([f'{name} ({cat}, {area:.2f})' for name, cat, area in train_stats['anomaly_large_bboxes']]) or 'None'}, Val={', '.join([f'{name} ({cat}, {area:.2f})' for name, cat, area in val_stats['anomaly_large_bboxes']]) or 'None'}\n\n")
        f.write(f"- Images with high class count (>30 of one class): Train={', '.join([f'{name} ({cls}, {count})' for name, cls, count in train_stats['anomaly_high_class_count']]) or 'None'}, Val={', '.join([f'{name} ({cls}, {count})' for name, cls, count in val_stats['anomaly_high_class_count']]) or 'None'}\n\n")

        f.write("\n## Patterns: Class Co-occurrences (Train)\n")
        for pair, cnt in sorted(train_stats['class_co_occurrences'].items(), key=lambda x: x[1], reverse=True)[:10]:
            f.write(f"- {pair}: {cnt}\n")

        f.write("\n## Patterns: Class Co-occurrences (Val)\n")
        for pair, cnt in sorted(val_stats['class_co_occurrences'].items(), key=lambda x: x[1], reverse=True)[:10]:
            f.write(f"- {pair}: {cnt}\n")

        f.write("\n## Visualizations\n")
        f.write("See plots in output/plots/<category>/ for charts (e.g., class_distribution/class_dist_train.png).\n")
        

def main():
    """Main function to run analysis."""
    # Paths to train  and val JSON files and output directory
    # train_path = '/app/data/bdd100k/labels/bdd100k_labels_images_train.json'
    # val_path = '/app/data/bdd100k/labels/bdd100k_labels_images_val.json'
    # output_dir = '/app/output'
    train_labels_path = 'data/bdd100k/labels/bdd100k_labels_images_train.json'
    val_labels_path = 'data/bdd100k/labels/bdd100k_labels_images_val.json'
    train_images_path = 'data/bdd100k/images/train'
    val_images_path = 'data/bdd100k/images/val'
    output_dir = 'output'
    plots_dir = os.path.join(output_dir, 'plots')
    report_file = os.path.join(output_dir, 'analysis_report.md')

    os.makedirs(output_dir, exist_ok=True)
    
    # Images count
    total_train_image_count = len(os.listdir(train_images_path))
    total_val_image_count = len(os.listdir(val_images_path))
    print(f"Train images found: {total_train_image_count}")
    print(f"Val images found: {total_val_image_count}")

    train_images, empty_train_images = filter_detection_labels(load_json_data(train_labels_path))
    val_images, empty_val_images = filter_detection_labels(load_json_data(val_labels_path))

    train_stats = compute_basic_stats(train_images, 'train', empty_train_images, total_train_image_count)
    val_stats = compute_basic_stats(val_images, 'val', empty_val_images, total_val_image_count)

    generate_plots(train_stats, plots_dir)
    generate_plots(val_stats, plots_dir)

    write_report(train_stats, val_stats, report_file)
    print(f"Analysis complete. Report saved to {report_file}")

if __name__ == "__main__":
    main()