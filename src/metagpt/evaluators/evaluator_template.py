"""
This module contains the EvaluatorTemplate class, which is responsible for evaluating
the performance of OME XML generation models by calculating the edit distance between
the ground truth and the prediction.

The class provides various methods for data analysis and visualization, including
edit distance calculations, path analysis, and performance comparisons across
different methods and image formats.
"""

import os
import copy
import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from deprecated import deprecated
from openai import OpenAI
from pydantic import BaseModel
from pygram.PyGram import Profile
from zss import simple_distance, Node
from ome_types import from_xml, to_xml, to_dict, OME
from ome_types.model import StructuredAnnotations

from metagpt.utils import utils
from metagpt.utils.DataClasses import Dataset
import base64
class EvaluatorTemplate:
    """
    This class evaluates the performance of an OME XML generation model by calculating
    the edit distance between the ground truth and the prediction.

    Reference: https://github.com/timtadh/zhang-shasha
    """

    def __init__(self, schema: Optional[str] = None, dataset: Optional[Dataset] = None, out_path: Optional[str] = None):
        """
        Initialize the EvaluatorTemplate.

        Args:
            schema (Optional[str]): The schema to use for evaluation.
            dataset (Optional[Dataset]): The dataset to evaluate.
            out_path (Optional[str]): The output path for saving results.
        """
        self.all_paths = None
        self.gt_graph = None
        self.pred_graph = None
        self.out_path = out_path
        self.scatter_size = 50
        self.font_size = 16
        self.figsize = (8, 5)
        self.x_tick_size = 12
        self.edit_score = []
        self.dataset = dataset
        self.plot_dict = {}

        self._create_output_directories()
        self._setup_color_palettes()

    def _create_output_directories(self):
        """Create necessary output directories."""
        for directory in ['plots', 'reports', 'data_frames']:
            os.makedirs(os.path.join(self.out_path, directory), exist_ok=True)

    def _setup_color_palettes(self):
        """Set up color palettes for plotting."""
        self.qual_color_palette = sns.color_palette("husl", 8)
        light_grey = (0.8, 0.8, 0.8)
        dark_grey = (0.4, 0.4, 0.4)
        self.palette0 = sns.palettes._ColorPalette(sns.color_palette("Paired")[0::2])
        self.palette0_bf = sns.palettes._ColorPalette([light_grey] + self.palette0)
        self.palette1 = sns.palettes._ColorPalette(sns.color_palette("Paired")[1::2])
        self.palette1_bf = sns.palettes._ColorPalette([dark_grey] + self.palette1)

    def json_to_pygram(self, json_data: Dict[str, Any]) -> Any:
        """
        Convert a JSON structure to a pygram tree.

        Args:
            json_data (Dict[str, Any]): The JSON data to convert.

        Returns:
            Any: The root node of the pygram tree.
        """
        def convert_element(key, value):
            node = Node(key)
            if isinstance(value, dict):
                for k, v in value.items():
                    node.addkid(convert_element(k, v))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    node.addkid(convert_element(f"{key}_{i}", item))
            else:
                node.addkid(Node(str(value)))
            return node

        root_key, root_value = next(iter(json_data.items()))
        return convert_element(root_key, root_value)

    def zss_edit_distance(self, xml_a: OME, xml_b: OME) -> int:
        """
        Calculate the Zhang-Shasha edit distance between two XML trees.

        Args:
            xml_a (OME): The first XML tree.
            xml_b (OME): The second XML tree.

        Returns:
            int: The edit distance between the two trees.
        """
        self.pred_graph = self.get_graph(xml_a)
        self.gt_graph = self.get_graph(xml_b)
        return simple_distance(self.gt_graph, self.pred_graph)

    def pygram_edit_distance(self, xml_a: OME, xml_b: OME) -> float:
        """
        Calculate the edit distance between two XML trees using pygram.

        Args:
            xml_a (OME): The first XML tree.
            xml_b (OME): The second XML tree.

        Returns:
            float: The edit distance between the two trees.
        """
        print("- - - Calculating Edit Distance - - -")
        json_a = {"ome": to_dict(xml_a)}
        json_b = {"ome": to_dict(xml_b)}
        profile1 = Profile(self.json_to_pygram(json_a), 2, 3)
        profile2 = Profile(self.json_to_pygram(json_b), 2, 3)
        return profile1.edit_distance(profile2)

    def get_graph(self, xml_root: OME, root: Optional[Node] = None) -> Node:
        """
        Get the graph representation of an OME XML tree as a zss Node.

        Args:
            xml_root (OME): The root of the XML tree.
            root (Optional[Node]): The root node of the graph (used for recursion).

        Returns:
            Node: The root node of the graph representation.
        """
        if root is None:
            root = Node("OME")
            if xml_root.attrib:
                for key, value in xml_root.attrib.items():
                    root.addkid(Node(f"{key}={value}"))

        for child in xml_root:
            new_node = Node(child.tag.split('}')[1])
            root.addkid(new_node)
            if child.attrib:
                for key, value in child.attrib.items():
                    new_node.addkid(Node(f"{key}={value}"))
            self.get_graph(child, new_node)
        return root

    def path_difference(self, xml_a: OME, xml_b: OME) -> int:
        """
        Calculate the length of the difference between the path sets in two XML trees.

        Args:
            xml_a (OME): The first XML tree.
            xml_b (OME): The second XML tree.

        Returns:
            int: The length of the difference between the path sets.
        """
        paths_a = utils.get_json(xml_a)
        paths_b = utils.get_json(xml_b)
        return len(paths_a.symmetric_difference(paths_b))

    def generate_results_report(self, figure_paths: List[str], context: str) -> Optional[str]:
        """
        Generate a results report based on the provided figures and context.

        Args:
            figure_paths (List[str]): Paths to the figure images.
            context (str): Context information for the report.

        Returns:
            Optional[str]: The generated report, or None if an error occurred.
        """
        client = OpenAI()
        base64_images = []
        
        for path in figure_paths[:9]:  # Limit to 9 images
            with open(path, "rb") as image_file:
                base64_images.append(base64.b64encode(image_file.read()).decode("utf-8"))

        messages = [
            {
                "role": "system",
                "content": """
                You are a scientific report generator. Your task is to analyze the provided figures and context, 
                and generate a formal, concise, and scientific results report. The report should:
                1. Describe the key findings shown in the figures
                2. Relate these findings to the context provided
                3. Use precise scientific language and maintain an objective tone
                4. Be concise yet comprehensive
                5. Avoid speculation beyond what is directly supported by the data
                Furthermore, you should respond using markdown syntax.
                Importantly, structure the report using sections and subsections as appropriate.
                The highest level section should be "Results" and the subsections should be based on the content of the report.
                Embed the figures in the report and refer to them in the text.
                Don't forget scientific captions for the figures.
                You should include each figure.
                """
            },
            {
                "role": "user",
                "content": [
                    f"Context: {context}\n\nPlease analyze the following figures and generate a results report. The paths to the images are {figure_paths}",
                    *[{"image": img, "resize": 768} for img in base64_images],
                ],
            },
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=messages,
                max_tokens=4096,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def report(self):
        """
        Generate and write an evaluation report to a file.
        """
        with open(f"/home/aaron/Documents/Projects/MetaGPT/out/context", "r") as f:
            context = f.read()

        df_paths = self.path_df()
        df_sample = self.sample_df(df_paths)
        
        self._generate_plots(df_sample)

        with open(f"{self.out_path}/reports/report_test.md", "w") as f:
            figure_paths = [os.path.join(self.out_path, "plots", file) 
                            for file in os.listdir(os.path.join(self.out_path, "plots")) 
                            if file.endswith(".png")]
            report = None # self.generate_results_report(figure_paths, context)
            if report:
                f.write(report)

    def _generate_plots(self, df_sample: pd.DataFrame):
        """Generate all plots."""
        plot_methods = [
            self.method_edit_distance_plt,
            self.n_paths_method_plt,
            self.format_method_plt,
            self.paths_annotation_stacked_plt,
            self.method_attempts_plt,
            self.format_counts_plt,
            self.attempts_paths_plt,
            self.method_edit_distance_no_annot_plt,
            self.method_edit_distance_only_annot_plt,
            self.method_cost_plt,
            self.method_time_plt,
            self.n_paths_cost_plt,
            self.n_paths_time_plt,
            self.plot_price_per_token
        ]

        for method in plot_methods:
            method(df_sample)

    def sample_df(self, df_paths: pd.DataFrame) -> pd.DataFrame:
        """
        Create a DataFrame with samples as Index and properties as Columns.

        Args:
            df_paths (pd.DataFrame): DataFrame containing path information.

        Returns:
            pd.DataFrame: DataFrame with sample properties.
        """
        properties = ["method", "n_paths", "n_annotations", "edit_distance"]
        df = pd.DataFrame(index=df_paths.columns, columns=properties)
        
        for prop in ["method", "name", "file_name", "format", "cost", "time", "index", "attempts"]:
            df[prop] = [getattr(s, prop, None) for s in self.dataset.samples.values()]
        df["n_paths"] = df_paths.sum()
        bioformats_mapping = df[df["method"] == "Bioformats"].groupby("file_name")["n_paths"].first()
        df["og_n_paths"] = df["file_name"].map(bioformats_mapping)
        df["n_annotations"] = {k: df_paths[k][df_paths.index.str.contains("structured_annotations")].sum() for k in df_paths.columns}
        df["og_image_format"] = df["format"]
        
        edit_distances = self._calculate_edit_distances(df)
        df["edit_distance"] = edit_distances["full"]
        df["edit_distance_no_annot"] = edit_distances["no_annot"]
        df["edit_distance_only_annot"] = edit_distances["only_annot"]

        df.to_csv(f"{self.out_path}/data_frames/sample_df.csv")
        return df

    def _calculate_edit_distances(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Calculate edit distances for full, no annotations, and only annotations."""
        distances = {"full": [], "no_annot": [], "only_annot": []}
        
        for name in df["name"].unique():
            gt_key = f"{df['file_name'][df['name']==name].values[0]}_Bioformats_{str(df['index'][df['name']==name].values[0])}"
            gt = self.dataset.samples[gt_key].metadata_xml
            test = self.dataset.samples[name].metadata_xml or None
            if test and gt:
                distances["full"].append(self.pygram_edit_distance(test, gt))
                
                gt_no_annot = copy.deepcopy(gt)
                gt_no_annot.structured_annotations = StructuredAnnotations()
                test_no_annot = copy.deepcopy(test)
                test_no_annot.structured_annotations = StructuredAnnotations()
                distances["no_annot"].append(self.pygram_edit_distance(test_no_annot, gt_no_annot))
                
                distances["only_annot"].append(self.pygram_edit_distance(test.structured_annotations, gt.structured_annotations))
            else:
                distances["full"].append(1)
                distances["no_annot"].append(1)
                distances["only_annot"].append(1)
        
        return distances
    
    def path_df(self) -> pd.DataFrame:
        """
        Create a DataFrame with paths as Index and samples as Columns.

        Returns:
            pd.DataFrame: DataFrame with path information.
        """
        print("- - - Creating Path DataFrame - - -")
        for s in self.dataset.samples.values():
            if s.metadata_str:
                s.metadata_xml = from_xml(s.metadata_str)
                s.paths = utils.generate_paths(to_dict(s.metadata_xml))
            else:
                s.paths = set()
            pd.Series(s.paths).to_csv(f"{self.out_path}/data_frames/path_df_{s.name+s.method}.csv")

        self.all_paths = pd.Series(list(set(utils.flatten([s.paths for s in self.dataset.samples.values()]))))
        
        df = pd.DataFrame(columns=[s.name for s in self.dataset.samples.values()], index=self.all_paths)
        
        for name, path in zip([s.name for s in self.dataset.samples.values()], [s.paths for s in self.dataset.samples.values()]):
            df[name] = df.index.isin(path)

        df.to_csv(f"{self.out_path}/data_frames/path_df.csv")
        return df

    def _create_base_plot(self, title: str) -> Tuple[plt.Figure, plt.Axes]:
        """Create a base plot with common settings."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlabel("Prediction Method", fontsize=self.font_size)
        ax.set_ylabel(title, fontsize=self.font_size)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return fig, ax

    def _save_plot(self, fig: plt.Figure, plot_name: str):
        """Save the plot as SVG and PNG."""
        plt.tight_layout()
        svg_path = f"{self.out_path}/plots/{plot_name}.svg"
        png_path = f"{self.out_path}/plots/{plot_name}.png"
        fig.savefig(svg_path)
        fig.savefig(png_path)
        self.plot_dict[plot_name] = f"../plots/{plot_name}.svg"

    def method_edit_distance_plt(self, df_sample: pd.DataFrame):
        """Plot method edit distance."""
        fig, ax = self._create_base_plot("Edit Distance")
        df = df_sample[df_sample["method"] != "Bioformats"]
        sns.barplot(x=df["method"], y=df["edit_distance"], edgecolor='black', ax=ax, palette=self.palette0)
        self._format_x_axis(ax, df["method"].unique())
        self._save_plot(fig, "method_edit_distance_plt")

    def n_paths_method_plt(self, df_sample: pd.DataFrame):
        """Plot number of paths per method."""
        fig, ax = self._create_base_plot("Number of Paths")
        sns.barplot(x=df_sample["method"], y=df_sample["n_paths"], edgecolor='black', ax=ax, palette=self.palette0_bf)
        self._format_x_axis(ax, df_sample["method"].unique())
        self._save_plot(fig, "n_paths_method_plt")

    def format_method_plt(self, df_sample: pd.DataFrame):
        """Plot edit distance by method and image format."""
        fig, ax = self._create_base_plot("Edit Distance")
        df = df_sample[df_sample["method"] != "Bioformats"]
        sns.barplot(x=df["method"], y=df["edit_distance"], hue=df_sample["og_image_format"], edgecolor='black', ax=ax, palette=self.palette0)
        self._format_x_axis(ax, df["method"].unique())
        ax.legend(loc='upper right')
        self._save_plot(fig, "format_method_plt")

    def paths_annotation_stacked_plt(self, df_sample: pd.DataFrame):
        """Plot stacked bar chart of paths and annotations."""
        fig, ax = self._create_base_plot("Number of Paths")
        n_path_plt = sns.barplot(x=df_sample["method"], y=df_sample["n_paths"]-df_sample["n_annotations"], edgecolor='black', ax=ax, palette=self.palette0_bf, label="OME Paths")
        bottom_heights = [patch.get_height() for patch in n_path_plt.patches]
        sns.barplot(x=df_sample["method"], y=df_sample["n_annotations"], edgecolor='black', ax=ax, palette=self.palette1_bf, label="Annotation Paths", bottom=bottom_heights)
        self._format_x_axis(ax, df_sample["method"].unique())
        ax.legend(loc='upper right')
        self._save_plot(fig, "paths_annotation_stacked_plt")

    def method_attempts_plt(self, df_sample: pd.DataFrame):
        """Plot number of attempts by method."""
        fig, ax = self._create_base_plot("Number of Attempts")
        df = df_sample[df_sample["method"] != "Bioformats"]
        sns.barplot(x=df["method"], y=df["attempts"], edgecolor='black', ax=ax, palette=self.palette0)
        self._format_x_axis(ax, df["method"].unique())
        self._save_plot(fig, "method_attempts_plt")

    def format_counts_plt(self, df_sample: pd.DataFrame):
        """Plot counts of samples by image format."""
        fig, ax = self._create_base_plot("Number of Samples")
        y = df_sample[df_sample["method"] == "Bioformats"]["og_image_format"].value_counts()
        sns.barplot(x=y.index, y=y, edgecolor='black', ax=ax, palette=self.qual_color_palette)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        self._save_plot(fig, "format_counts_plt")

    def attempts_paths_plt(self, df_sample: pd.DataFrame):
        """Plot number of attempts against number of paths."""
        fig, ax = self._create_base_plot("Number of Attempts")
        df = df_sample[df_sample["method"] != "Bioformats"]
        noise = np.random.normal(-0.1, 0.1, size=len(df))
        sns.regplot(data=df, x="og_n_paths", y="attempts", order=1, color=".3")
        sns.scatterplot(x=df["og_n_paths"], y=df["attempts"]+noise, hue=df["method"], ax=ax, palette=self.palette0, alpha=0.9, s=self.scatter_size)
        ax.set_xlabel("Number of Paths", fontsize=self.font_size)
        ax.legend(loc='upper left')
        self._save_plot(fig, "attempts_paths_plt")

    def method_edit_distance_no_annot_plt(self, df_sample: pd.DataFrame):
        """Plot method edit distance without annotations."""
        fig, ax = self._create_base_plot("Edit Distance (No Annotations)")
        df = df_sample[df_sample["method"] != "Bioformats"]
        sns.barplot(x=df["method"], y=df["edit_distance_no_annot"], edgecolor='black', ax=ax, palette=self.palette0)
        self._format_x_axis(ax, df["method"].unique())
        self._save_plot(fig, "method_edit_distance_no_annot_plt")

    def method_edit_distance_only_annot_plt(self, df_sample: pd.DataFrame):
        """Plot method edit distance for annotations only."""
        fig, ax = self._create_base_plot("Edit Distance (Annotations Only)")
        df = df_sample[df_sample["method"] != "Bioformats"]
        sns.barplot(x=df["method"], y=df["edit_distance_only_annot"], edgecolor='black', ax=ax, palette=self.palette0)
        self._format_x_axis(ax, df["method"].unique())
        self._save_plot(fig, "method_edit_distance_only_annot_plt")

    def method_cost_plt(self, df_sample: pd.DataFrame):
        """Plot method cost."""
        fig, ax = self._create_base_plot("Cost in $")
        df = df_sample[df_sample["method"] != "Bioformats"]
        sns.barplot(x=df["method"], y=df["cost"], edgecolor='black', ax=ax, palette=self.palette0)
        self._format_x_axis(ax, df["method"].unique())
        self._save_plot(fig, "method_cost_plt")

    def method_time_plt(self, df_sample: pd.DataFrame):
        """Plot method prediction time."""
        fig, ax = self._create_base_plot("Prediction Time in s")
        sns.barplot(x=df_sample["method"], y=df_sample["time"], edgecolor='black', ax=ax, palette=self.palette0_bf)
        self._format_x_axis(ax, df_sample["method"].unique())
        self._save_plot(fig, "method_time_plt")

    def n_paths_cost_plt(self, df_sample: pd.DataFrame):
        """Plot number of paths against cost."""
        fig, ax = self._create_base_plot("Cost in $")
        df = df_sample[df_sample["method"] != "Bioformats"]
        sns.regplot(data=df, x="og_n_paths", y="cost", order=1, color=".3")
        sns.scatterplot(x=df["og_n_paths"], y=df["cost"], hue=df["method"], ax=ax, palette=self.palette0, s=self.scatter_size)
        ax.set_xlabel("Number of Paths", fontsize=self.font_size)
        ax.legend(loc='upper right')
        self._save_plot(fig, "n_paths_cost_plt")

    def n_paths_time_plt(self, df_sample: pd.DataFrame):
        """Plot number of paths against prediction time."""
        fig, ax = self._create_base_plot("Time in s")
        df = df_sample[df_sample["method"] != "Bioformats"]
        sns.regplot(data=df, x="og_n_paths", y="time", order=1, color=".3")
        sns.scatterplot(x=df["og_n_paths"], y=df["time"], hue=df_sample["method"], ax=ax, palette=self.palette0, s=self.scatter_size)
        ax.set_xlabel("Number of Paths", fontsize=self.font_size)
        ax.legend(loc='upper right')
        self._save_plot(fig, "n_paths_time_plt")

    def plot_price_per_token(self, method):
        """Plot price per token for different models."""
        class PriceDevelopments(BaseModel):
            model: str
            price_in: float
            price_out: float
            release_date: datetime.date
            mmlu: float

        dataset = [PriceDevelopments(model="GPT-4-turbo", mmlu=85.5, price_in=10, price_out=30, release_date=datetime.date(year=2023, month=11, day=6)),
                PriceDevelopments(model="GPT-4o",mmlu=88.7, price_in=5, price_out=15, release_date=datetime.date(year=2024, month=5, day=13)),
                PriceDevelopments(model="GPT-4o-mini",mmlu=82.0, price_in=0.15, price_out=0.6, release_date=datetime.date(year=2024, month=7, day=18)),
                PriceDevelopments(model="GPT-4",mmlu=85.4, price_in=30, price_out=60, release_date=datetime.date(year=2023, month=3, day=14)),
                PriceDevelopments(model="GPT-3.5-turbo",mmlu=70.0, price_in=0.5, price_out=1.5, release_date=datetime.date(year=2022, month=11, day=28)),
                PriceDevelopments(model="Gemini-1.5-pro",mmlu=85.9, price_in=7, price_out=21, release_date=datetime.date(year=2024, month=5, day=1)),
                PriceDevelopments(model="Gemini-1.5-pro",mmlu=81.9, price_in=7, price_out=21, release_date=datetime.date(year=2024, month=2, day=1)),
                PriceDevelopments(model="Gemini-1.5-flash",mmlu=78.9, price_in=0.13, price_out=0.38, release_date=datetime.date(year=2024, month=5, day=14)),
                PriceDevelopments(model="Claude-2", mmlu=78.5, price_in=8, price_out=24, release_date=datetime.date(year=2023, month=7, day=11)),
                PriceDevelopments(model="Claude-3-opus", mmlu=86.8, price_in=15, price_out=75, release_date=datetime.date(year=2024, month=3, day=4)),
                PriceDevelopments(model="Claude-3-sonnet", mmlu=79, price_in=3, price_out=15, release_date=datetime.date(year=2024, month=3, day=4)),
                PriceDevelopments(model="Claude-3-haiku", mmlu=75.2, price_in=0.25, price_out=1.25, release_date=datetime.date(year=2024, month=3, day=13)),
                PriceDevelopments(model="Claude-3.5-sonnet", mmlu=88.7, price_in=3, price_out=15, release_date=datetime.date(year=2024, month=6, day=20)),
                PriceDevelopments(model="Mistral-8x7b", mmlu=70.6, price_in=0.7, price_out=0.7, release_date=datetime.date(year=2023, month=12, day=1)),
                PriceDevelopments(model="Mistral-large", mmlu=81.2, price_in=8, price_out=8, release_date=datetime.date(year=2024, month=2, day=26)),
                PriceDevelopments(model="Mistral-large2",mmlu=84.0, price_in=3, price_out=9, release_date=datetime.date(year=2024, month=7, day=24)),
                PriceDevelopments(model="Mistral-7b-instruct",mmlu=60.1, price_in=0.25, price_out=0.25, release_date=datetime.date(year=2023, month=9, day=27)),
        ]

        df = pd.DataFrame([m.dict() for m in dataset])
        df["price_per_mmlu"] = df["price_in"] / df["mmlu"]
        df["model_family"] = df["model"].str.split("-").str[0]

        fig, ax = self._create_base_plot("Price per MMLU")
        reference_date = datetime.date(1970, 1, 1)
        df['release_days'] = (df['release_date'] - reference_date).apply(lambda x: x.days)

        sns.regplot(data=df, x="release_days", y="price_per_mmlu", order=1, color=".3")
        sns.scatterplot(data=df, x="release_date", y="price_per_mmlu", hue="model_family", s=self.scatter_size)

        plt.xlabel("Release Date", fontsize=self.font_size)
        plt.xticks(rotation=45)
        ax.legend(loc='upper right')
        self._save_plot(fig, "plot_price_per_token")

    def _format_x_axis(self, ax: plt.Axes, methods: List[str]):
        """Format x-axis labels."""
        methods_with_linebreaks = [method.replace("_", "\n") for method in methods]
        ax.set_xticklabels(methods_with_linebreaks, rotation=0, ha='center', fontsize=self.x_tick_size)
        plt.yticks(fontsize=self.x_tick_size)