import time
import xml.etree.ElementTree as ET
from typing import Union, Any
import numpy as np
from zss import simple_distance, Node
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from metagpt.utils.DataClasses import Sample, Dataset
import pygram.tree as tree
from pygram.PyGram import Profile
from deprecated import deprecated
from metagpt.utils import utils
from ome_types import from_xml, to_xml, to_dict, OME
from ome_types.model import StructuredAnnotations
import copy
import ast
import openai
import os
from openai import OpenAI
import base64
import cv2
from pydantic import BaseModel, Field
import datetime

class EvaluatorTemplate:
    """
    This class evaluates the performance of a OME XML generation model by calculating the edit distance between the
    ground truth and the prediction. https://github.com/timtadh/zhang-shasha
    """

    def __init__(self,
                 schema: str = None,
                 dataset: Dataset = None,
                 out_path: str = None):
        """
        :param path_to_raw_metadata: path to the raw metadata file
        """
        self.all_paths = None
        self.gt_graph = None
        self.pred_graph = None
        self.out_path: str = out_path
        self.scatter_size = 50
        self.font_size = 16
        self.figsize = (8, 5)
        self.x_tick_size = 12
        if not os.path.exists(os.path.dirname(f"{self.out_path}/plots/")):
            os.makedirs(os.path.dirname(f"{self.out_path}/plots/"))

        if not os.path.exists(os.path.dirname(f"{self.out_path}/reports/")):
            os.makedirs(os.path.dirname(f"{self.out_path}/reports/"))

        if not os.path.exists(os.path.dirname(f"{self.out_path}/data_frames/")):
            os.makedirs(os.path.dirname(f"{self.out_path}/data_frames/"))
        self.edit_score = []
        self.dataset = dataset
        self.plot_dict = {}
        self.qual_color_palette = sns.color_palette("husl", 8)
        # Define the greyish color
        light_grey = (0.8, 0.8, 0.8)  # RGB values for a greyish color
        dark_grey = (0.4, 0.4, 0.4)  # RGB values for a greyish color
        # Prepend the greyish color to the palette
        self.palette0 = sns.palettes._ColorPalette(sns.color_palette("Paired")[0::2])
        self.palette0_bf = sns.palettes._ColorPalette([light_grey] + self.palette0)
        self.palette1 = sns.palettes._ColorPalette(sns.color_palette("Paired")[1::2])
        self.palette1_bf = sns.palettes._ColorPalette([dark_grey] + self.palette1)

        self.time:str

    def json_to_pygram(self, json_data: dict):
        """
        Convert a JSON structure to a pygram tree.
        """
        def convert_element(key, value):
            node = tree.Node(key)
            
            if isinstance(value, dict):
                for k, v in value.items():
                    node.addkid(convert_element(k, v))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    node.addkid(convert_element(f"{key}_{i}", item))
            else:
                node.addkid(tree.Node(str(value)))
            
            return node

        # Assuming the input is a dictionary with a single root element
        root_key, root_value = next(iter(json_data.items()))
        return convert_element(root_key, root_value)
    

    def zss_edit_distance(self, xml_a: ET.Element, xml_b: ET.Element):
        """
        TODO: add docstring
        """
        self.pred_graph = self.get_graph(xml_a)
        self.gt_graph = self.get_graph(xml_b)
        return simple_distance(self.gt_graph, self.pred_graph)
    

    def pygram_edit_distance(self, xml_a: OME, xml_b: OME):
        """
        Calculate the edit distance between two xml trees on word level.
        Here an outline of the algorithm:
        """
        print("- - - Calculating Edit Distance - - -")
        json_a = {"ome": to_dict(xml_a)}
        json_b = {"ome": to_dict(xml_b)}
        profile1 = Profile(self.json_to_pygram(json_a), 2, 3)
        profile2 = Profile(self.json_to_pygram(json_b), 2, 3)
        return profile1.edit_distance(profile2)
    

    @deprecated()
    def word_edit_distance(self, aligned_paths) -> int:
        """
        Calculate the word level edit distance between two sets of paths.
        aligned_paths: list of tuples of aligned paths
        """
        edit_distance = 0
        for path_a, path_b in aligned_paths:
            # set the number of iterations to the length of the longest path
            iterations = len(path_a) if len(path_a) > len(path_b) else len(path_b)
            distance = 0
            for i in range(iterations):
                node_b = path_b[i] if len(path_b) > i else None
                node_a = path_a[i] if len(path_a) > i else None
                if node_a != node_b:
                    distance += 1
            edit_distance += distance
        return edit_distance
    

    @deprecated()
    def align_paths(self, paths_a, paths_b):
        """
        Align the paths such that the sum of distances between the paths is minimized.
        paths_a: set of paths
        paths_b: set of paths
        :return: list of tuples of aligned paths
        """
        print("- - - Aligning paths - - -")
        # Assuming A and B are your lists and f is your function

        # Create the matrix
        matrix = [[self.align_sequences_score(a, b) for b in paths_b] for a in paths_a]
        print("matrix", [print(x, "\n") for x in matrix])

        # score, alignment_a, alignment_b = self.align_sequences(paths_a, paths_b, cost=self.align_sequences_score)
        return 1, 1, 1

    @deprecated()
    def align_sequences_score(self, s1, s2, cost=lambda a, b: a != b):
        """
        returns only the score for the alignment
        :param s1:
        :param s2:
        :param cost:
        :return:
        """
        score, alignment_a, alignment_b = self.align_sequences(s1, s2, cost=lambda a, b: a != b)
        return score

    @deprecated()
    def align_sequences(self, s1, s2, cost=lambda a, b: a != b):
        print("- - - Aligning sequences - - -")
        m, n = len(s1), len(s2)
        # Create a matrix to store the cost of edits
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        dp = np.zeros((m + 1, n + 1))

        # Initialize the matrix with the cost of deletions and insertions
        for i in range(m + 1):
            dp[i][0] = i * +1
        for j in range(n + 1):
            dp[0][j] = j * +1

        # Fill the matrix based on the cost of substitution, insertion, and deletion
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]  # No cost if characters are the same
                else:
                    dp[i][j] = min(dp[i - 1][j - 1] + cost(s1[i - 1], s2[j - 1]),  # Substitution
                                   dp[i - 1][j] + cost(s1[i - 1], s2[j - 1]),  # Deletion
                                   dp[i][j - 1] + cost(s1[i - 1], s2[j - 1]),  # Insertion
                                   )
        # Reconstruct the alignment
        alignment_a, alignment_b = [], []
        i, j = m, n
        while i > 0 and j > 0:
            if s1[i - 1] == s2[j - 1]:
                alignment_a = [s1[i - 1]] + alignment_a
                alignment_b = [s2[j - 1]] + alignment_b
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i - 1][j - 1] + cost(s1[i - 1], s2[j - 1]):  # substitution
                alignment_a = [s1[i - 1]] + alignment_a
                alignment_b = [s2[j - 1]] + alignment_b
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i - 1][j] + cost(s1[i - 1], s2[j - 1]):  # deletion
                alignment_a = [s1[i - 1]] + alignment_a
                alignment_b = [["-"]] + alignment_b
                i -= 1
            else:  # dp[i][j] == dp[i][j - 1] + cost_insertion  # insertion
                alignment_a = [["-"]] + alignment_a
                alignment_b = [s2[j - 1]] + alignment_b
                j -= 1

        # Handle the remaining characters in s1 or s2
        while i > 0:
            alignment_a = [s1[i - 1]] + alignment_a
            alignment_b = [[["-"] * len(s1[j - 1])]] + alignment_b
            i -= 1
        while j > 0:
            alignment_a = [[["-"] * len(s2[j - 1])]] + alignment_a
            alignment_b = [s2[j - 1]] + alignment_b
            j -= 1
        return dp[-1][-1], alignment_a, alignment_b
    

    def get_graph(self, xml_root: ET.Element, root=None)-> Node:
        """
        Helper function to get the graph representation of an ET XML tree as zss Node.
        """
        if root is None:
            root = Node("OME")
            if xml_root.attrib:
                for key in xml_root.attrib.keys():
                    root.addkid(Node(key + '=' + xml_root.attrib[key]))

        for child in xml_root:
            new_node = Node(child.tag.split('}')[1])
            root.addkid(new_node)
            if child.attrib:
                for key in child.attrib.keys():
                    new_node.addkid(Node(key + '=' + child.attrib[key]))
            self.get_graph(child, new_node)
        return root
    

    def path_difference(self, xml_a: ET.Element, xml_b: ET.Element):
        """
        Calculates the length of the difference between the path sets in two xml trees.
        """
        paths_a = utils.get_json(xml_a)
        paths_b = utils.get_json(xml_b)
        return len(paths_a.symmetric_difference(paths_b))
    

    def generate_results_report(self, figure_paths: list, context: str):
        # Initialize the OpenAI client
        client = OpenAI()

        # Encode images to base64
        base64_images = []
        if len(figure_paths) > 9:
            figure_paths = figure_paths[:9]
        for path in figure_paths:  # Limit to 9 images
            img = cv2.imread(path)
            _, buffer = cv2.imencode(".png", img)
            base64_images.append(base64.b64encode(buffer).decode("utf-8"))

        # Prepare the messages
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
                    *map(lambda x: {"image": x, "resize": 768}, base64_images),
                ],
            },
        ]

        # Call the API
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
            )

            # Extract the report from the response
            report = response.choices[0].message.content

            return report
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def report(self):
        """
        Write an evaluation report to file.
        """
        with open(f"/home/aaron/Documents/Projects/MetaGPT/out/context", "r") as f:
            context = f.read()

        with open(f"{self.out_path}/reports/report_test.md", "w") as f:
            # create a dataframe with the paths
            df_paths = self.path_df()
            # create a dataframe with the samples properties
            df_sample = self.sample_df(df_paths)
            # create the plots
            self.method_edit_distance_plt(df_sample)
            self.n_paths_method_plt(df_sample)
            self.format_method_plt(df_sample)
            self.paths_annotation_stacked_plt(df_sample)
            self.method_attempts_plt(df_sample)
            self.format_counts_plt(df_sample)
            self.attempts_paths_plt(df_sample)
            self.method_edit_distance_no_annot_plt(df_sample)
            self.method_edit_distance_only_annot_plt(df_sample)
            self.method_cost_plt(df_sample)
            self.method_time_plt(df_sample)
            self.n_paths_cost_plt(df_sample)
            self.n_paths_time_plt(df_sample)
            self.plot_price_per_token()
            #figure_paths = [f"{self.out_path}/plots/{file}" for file in os.listdir(f"{self.out_path}/plots") if file.endswith(".png")]
            #print(figure_paths)
            #report = self.generate_results_report(figure_paths, context)
            #f.write(report)


    def sample_df(
            self,
            df_paths: pd.DataFrame = None,
    ):
        """
        This function creates a df with samples as Index and properties as Columns.
        TODO: Add docstring
        """
        properties = ["Method", "n_paths", "n_annotations", "Edit_distance"]
        df = pd.DataFrame(index=df_paths.columns, columns=properties)
        df["Method"] = [s.method if s.method else None for s in self.dataset.samples.values()]
        df["Name"] = [s.name if s.name else None for s in self.dataset.samples.values()]
        df["file_name"] = [s.file_name if s.file_name else None for s in self.dataset.samples.values()]
        df["n_paths"] = df_paths.sum()
        # get the number of paths for each sample depending on how many paths were in the input data(the bioformats path count)
        # Create a mapping of 'Name' to 'n_paths' for rows where 'Method' is 'Bioformats'
        bioformats_mapping = df[df["Method"] == "Bioformats"].groupby("file_name")["n_paths"].first()
        # Apply this mapping to the entire DataFrame
        df["og_n_paths"] = df["file_name"].map(bioformats_mapping)
        df["n_annotations"] = {k: df_paths[k][df_paths.index.str.contains("structured_annotations")].sum() for k in
                               df_paths.columns}
        df["og_image_format"] = [s.format if s.format else None for s in self.dataset.samples.values()]
        df["cost"] = [s.cost for s in self.dataset.samples.values()]
        df["time"] = [s.time for s in self.dataset.samples.values()]
        df["index"] = [s.index for s in self.dataset.samples.values()] # cant have the if else here because if index=0 is False .... 
        df["attempts"] = [s.attempts if s.attempts else None for s in self.dataset.samples.values()]
        
        edit_distances = [1]*len(df["Name"].unique())
        edit_distances_no_annot = [1]*len(df["Name"].unique())
        edit_distances_only_annot = [1]*len(df["Name"].unique())

        for j, n in enumerate(df["Name"].unique()):
            gt = self.dataset.samples[f"{df['file_name'][df['Name']==n].values[0]}_Bioformats_{str(df['index'][df['Name']==n].values[0])}"].metadata_xml
            gt_no_annot = copy.deepcopy(gt)
            gt_no_annot.structured_annotations = StructuredAnnotations()
            gt_only_annot = gt.structured_annotations
            test = self.dataset.samples[n].metadata_xml or None
            test_no_annot = copy.deepcopy(test)
            test_no_annot.structured_annotations = StructuredAnnotations()
            test_only_annot = test.structured_annotations
            if test and gt:
                edit_distances[j] = self.pygram_edit_distance(test, gt)
            if test_no_annot and gt_no_annot:
                edit_distances_no_annot[j] = self.pygram_edit_distance(test_no_annot, gt_no_annot)
            if test_only_annot and gt_only_annot:
                edit_distances_only_annot[j] = self.pygram_edit_distance(test_only_annot, gt_only_annot)

        df["Edit_distance"] = edit_distances
        df["Edit_distance_no_annot"] = edit_distances_no_annot
        df["Edit_distance_only_annot"] = edit_distances_only_annot

        # save the df to a csv
        df.to_csv(f"{self.out_path}/data_frames/sample_df.csv")

        return df


    def path_df(
            self,
    )-> pd.DataFrame:
        """
        This function creates a df with paths as Index and samples as Columns.
        The entries are True if the path is present in the sample and False if not.
        """
        print("- - - Creating Path DataFrame - - -")
        for s in self.dataset.samples.values():
            if s.metadata_str:
                #s.metadata_xml = ET.fromstring(s.metadata_str)
                s.metadata_xml = from_xml(s.metadata_str)
                s.paths = utils.generate_paths(to_dict(s.metadata_xml))
            else:
                s.paths = set()
            pd.Series(s.paths).to_csv(f"{self.out_path}/data_frames/path_df_{s.name+s.method}.csv")

        self.all_paths = pd.Series(
            list(
                set(
                    utils.flatten(
                        [s.paths for s in self.dataset.samples.values()]))))
        
        df = pd.DataFrame(
            columns=[f"{s.name}" for s in self.dataset.samples.values()],
            index=self.all_paths)
        
        for name, path in zip(
            [s.name for s in self.dataset.samples.values()],
            [s.paths for s in self.dataset.samples.values()]
            ):
            df[f"{name}"] = df.index.isin(path)

        #save the df to a csv
        df.to_csv(f"{self.out_path}/data_frames/path_df.csv")
        return df


    def method_edit_distance_plt(
            self,
            df_sample: pd.DataFrame = None,
    ):
        """
        This function creates a plot which compares the inter sample standard deviation.
        The X-axis will be the used method, whereas the Y-axis will be the standard deviation.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        df = df_sample[df_sample["Method"]!="Bioformats"]
        plot = sns.barplot(
            x=df["Method"], # all entries but bioformats
            y=df["Edit_distance"],
            edgecolor='black',
            ax=ax,
            palette=self.palette0)

        ax.set_xlabel("Prediction Method", fontsize=self.font_size)
        ax.set_ylabel("Edit Distance", fontsize=self.font_size)
        methods_with_linebreaks = [method.replace("_", "\n") for method in df["Method"].unique()]
        ax.set_xticklabels(methods_with_linebreaks, rotation=0, ha='center', fontsize=self.x_tick_size)
        plt.yticks(fontsize=self.x_tick_size)
        #ax.legend(loc='upper right')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        plt.savefig(f"{self.out_path}/plots/method_edit_distance_plt.svg")
        self.plot_dict["method_edit_distance_plt"] = f"../plots/method_edit_distance_plt.svg"
        # save as png
        plt.savefig(f"{self.out_path}/plots/method_edit_distance_plt.png")
        #plt.show()

        return fig, ax


    def paths_annotation_stacked_plt(
            self,
            df_sample: pd.DataFrame = None,
            ):
        """
        Plots the number of paths and annotations per sample as a stacked bar plot.
        Uses the seaborn library.

        """
        fig, ax = plt.subplots(figsize=self.figsize)

        n_path_plt = sns.barplot(
            x=df_sample["Method"],
            y=df_sample["n_paths"]-df_sample["n_annotations"],
            edgecolor='black',
            ax=ax,
            palette=self.palette0_bf,
            label="OME Paths")
        bottom_heights = [patch.get_height() for patch in n_path_plt.patches]
        n_annot_plt = sns.barplot(
            x=df_sample["Method"],
            y=df_sample["n_annotations"],
            edgecolor='black',
            ax=ax,
            palette=self.palette1_bf,
            label="Annotation Paths",
            bottom=bottom_heights)


        ax.set_xlabel("Prediction Method", fontsize=14)
        ax.set_ylabel("Number of Paths", fontsize=14)
        #ax.set_title("Number of Paths and Annotations per Sample", fontsize=16)
        methods_with_linebreaks = [method.replace("_", "\n") for method in df_sample["Method"].unique()]
        ax.set_xticklabels(methods_with_linebreaks, rotation=0, ha='center', fontsize=self.x_tick_size)
        plt.yticks(fontsize=self.x_tick_size)

        ax.legend(loc='upper right')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        plt.savefig(f"{self.out_path}/plots/paths_annotation_stacked_plt.svg")
        self.plot_dict["paths_annotation_stacked_plt"] = f"../plots/paths_annotation_stacked_plt.svg"
        # save as png
        plt.savefig(f"{self.out_path}/plots/paths_annotation_stacked_plt.png")
        #plt.show()

        return fig, ax
    
        
    def n_paths_method_plt(
            self, 
            df_sample: pd.DataFrame = None,
    ):
        """
        Plots the number of paths per method as a bar plot.
        
        Parameters:
        - df_sample: pd.DataFrame, a DataFrame containing the data to plot.
        
        Returns:
        - fig: The figure object.
        - ax: The axes object.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        plot = sns.barplot(
            x=df_sample["Method"],
            y=df_sample["n_paths"],
            edgecolor='black',
            ax=ax,
            palette=self.palette0_bf,
            label="OME Paths")


        ax.set_xlabel("Prediction Method", fontsize=self.font_size)
        ax.set_ylabel("Number of Paths", fontsize=self.font_size)
        #ax.set_title("Number of Paths and Annotations per Sample", fontsize=16)
        methods_with_linebreaks = [method.replace("_", "\n") for method in df_sample["Method"].unique()]
        ax.set_xticklabels(methods_with_linebreaks, rotation=0, ha='center', fontsize=self.x_tick_size)
        plt.yticks(fontsize=self.x_tick_size)

        ax.legend(loc='upper right')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        plt.savefig(f"{self.out_path}/plots/n_paths_method_plt.svg")
        #save as png
        plt.savefig(f"{self.out_path}/plots/n_paths_method_plt.png")
        self.plot_dict["n_paths_method_plt"] = f"../plots/n_paths_method_plt.svg"
        #plt.show()

        return fig, ax


    def format_method_plt(
            self,
            df_sample: pd.DataFrame = None,
    ):
        """
        This plot compares the performance of the different methods based on the original image format.
        For each method several bars are plotted, one for each image format.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        df = df_sample[df_sample["Method"]!="Bioformats"]

        plot = sns.barplot(
            x=df["Method"],
            y=df["Edit_distance"],
            hue=df_sample["og_image_format"],
            edgecolor='black',
            ax=ax,
            palette=self.palette0)


        ax.set_xlabel("Method", fontsize=self.font_size)
        ax.set_ylabel("Edit Distance", fontsize=self.font_size)
        methods_with_linebreaks = [method.replace("_", "\n") for method in df["Method"].unique()]
        ax.set_xticklabels(methods_with_linebreaks, rotation=0, ha='center', fontsize=self.x_tick_size)
        plt.yticks(fontsize=self.x_tick_size)

        ax.legend(loc='upper right')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        # save as svg
        plt.savefig(f"{self.out_path}/plots/format_method_plt.svg")
        # save as png
        plt.savefig(f"{self.out_path}/plots/format_method_plt.png")
        self.plot_dict["format_method_plt"] = f"../plots/format_method_plt.svg"
        #plt.show()

        return fig, ax
    
    def method_cost_plt(
            self,
            df_sample: pd.DataFrame = None,
    ):
        """
        This plot compares the performance of the different methods based on the cost.
        Wont work because OpenAI does not provide the cost of the methods. --> workarround: use the returned tokens as proxy.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        df = df_sample[df_sample["Method"]!="Bioformats"]
        plot = sns.barplot(
            x=df["Method"],
            y=df["cost"],
            edgecolor='black',
            ax=ax,
            palette=self.palette0)
        
        ax.set_xlabel("Method", fontsize=self.font_size)
        ax.set_ylabel("Cost in $", fontsize=self.font_size)
        methods_with_linebreaks = [method.replace("_", "\n") for method in df["Method"].unique()]
        ax.set_xticklabels(methods_with_linebreaks, rotation=0, ha='center', fontsize=self.x_tick_size)
        plt.yticks(fontsize=self.x_tick_size)


        #ax.legend(loc='upper right')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        plt.savefig(f"{self.out_path}/plots/method_cost_plt.svg")
        self.plot_dict["method_cost_plt"] = f"../plots/method_cost_plt.svg"
        #save as png
        plt.savefig(f"{self.out_path}/plots/method_cost_plt.png")
        #plt.show()
        return fig, ax

    def method_time_plt(self, df_sample: pd.DataFrame = None):
        """
        This plot compares the performance of the different methods based on the time it took to generate the OME XML.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Filter out methods with time 0 or None
        df_plt = df_sample
        
        # Create the barplot
        sns.barplot(
            x=df_plt["Method"],
            y=df_plt["time"], # TODO:add the bioformats time as this is the time to read in the raw data and applies to all methods
            edgecolor='black',
            ax=ax,
            palette=self.palette0_bf
        )
        
        ax.set_xlabel("Prediction Method", fontsize=self.font_size)
        ax.set_ylabel("Prediction Time in s", fontsize=self.font_size)    
        # Rotate x-axis labels for readability
        # add space between upper case letters
        methods_with_linebreaks = [method.replace("_", "\n") for method in df_sample["Method"].unique()]
        ax.set_xticklabels(methods_with_linebreaks, rotation=0, ha='center', fontsize=self.x_tick_size)
        plt.yticks(fontsize=self.x_tick_size)

        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()

        plt.savefig(f"{self.out_path}/plots/method_time_plt.svg")
        self.plot_dict["method_time_plt"] = f"../plots/method_time_plt.svg"
        #save as png
        plt.savefig(f"{self.out_path}/plots/method_time_plt.png")
        #plt.show()
        return fig, ax
    
    def method_attempts_plt(
            self,
            df_sample: pd.DataFrame = None,
    ):
        """
        This plots the number of attempts against the number of paths in the og image.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        df = df_sample[df_sample["Method"]!="Bioformats"]

        plot = sns.barplot(
            x=df["Method"],
            y=df["attempts"],
            edgecolor='black',
            ax=ax,
            palette=self.palette0)

        ax.set_xlabel("Number of Paths", fontsize=self.font_size)
        ax.set_ylabel("Number of Attempts", fontsize=self.font_size)
        methods_with_linebreaks = [method.replace("_", "\n") for method in df["Method"].unique()]
        ax.set_xticklabels(methods_with_linebreaks, rotation=0, ha='center', fontsize=self.x_tick_size)
        plt.yticks(fontsize=self.x_tick_size)

        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        plt.savefig(f"{self.out_path}/plots/method_attempts_plt.svg")
        self.plot_dict["method_attempts_plt"] = f"../plots/method_attempts_plt.svg"
        # save as png
        plt.savefig(f"{self.out_path}/plots/method_attempts_plt.png")
        #plt.show()

        return fig, ax
    
    def format_counts_plt(
            self,
            df_sample: pd.DataFrame = None,
    ):
        """
        This plot shows the formats on the x axis, and how many samples are in
        each format on the y axis. the different samples need to be identified via
        the name tag to not count the same file multiple times for each method.
        
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        y = df_sample[df_sample["Method"]=="Bioformats"]["og_image_format"].value_counts()
        plot = sns.barplot(
            x=y.index,
            y=y,
            edgecolor='black',
            ax=ax,
            palette=self.qual_color_palette)

        ax.set_xlabel("Format", fontsize=self.font_size)
        ax.set_ylabel("Number of Samples", fontsize=self.font_size)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        #ax.legend(loc='upper right')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        plt.savefig(f"{self.out_path}/plots/format_counts_plt.svg")
        self.plot_dict["format_counts_plt"] = f"../plots/format_counts_plt.svg"
        # save as png
        plt.savefig(f"{self.out_path}/plots/format_counts_plt.png")
        #plt.show()

        return fig, ax
    
    def format_n_paths_plt(
            self,
            df_sample: pd.DataFrame = None,
    ):
        """
        This plots shows the format on the x axis and the number of paths of the
        y axis as a scatter plot for each data point. But only for the Bioformats method.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        plot = sns.scatterplot(
            x=df_sample[df_sample["Method"]=="Bioformats"]["og_image_format"],
            y=df_sample[df_sample["Method"]=="Bioformats"]["n_paths"],
            edgecolor='black',
            ax=ax,
            palette=self.qual_color_palette,
            s=self.scatter_size)

        ax.set_xlabel("Format", fontsize=self.font_size)
        ax.set_ylabel("Number of Paths", fontsize=self.font_size)
        plt.yticks(fontsize=self.x_tick_size)
        plt.xticks(fontsize=self.x_tick_size)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        #ax.legend(loc='upper right')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        plt.savefig(f"{self.out_path}/plots/format_n_paths_plt.svg")
        self.plot_dict["format_n_paths_plt"] = f"../plots/format_n_paths_plt.svg"
        # save as png
        plt.savefig(f"{self.out_path}/plots/format_n_paths_plt.png")
        #plt.show()

        return fig, ax
    
    @deprecated()
    def paths_annotation_stacked_relative_plt(
            self,
            df_sample: pd.DataFrame = None,
            ):
        """
        Plots the relative to og_bioformats file number of paths and annotations
        per sample as a stacked bar plot.
        Uses the seaborn library.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        norm_values_n_paths = df_sample[df_sample['Method'] == 'Bioformats'].set_index('Name')['n_paths']

        n_path_plt = sns.barplot(
            x=df_sample["Method"],
            y=df_sample.apply(
            lambda row: (row['n_paths'] / norm_values_n_paths[row['Name']]),
            axis=1) * (df_sample["n_paths"]-df_sample["n_annotations"])/df_sample["n_paths"],
            edgecolor='black',
            ax=ax,
            label="OME Paths",
            palette=self.palette0_bf)

        bottom_heights = [patch.get_height() for patch in n_path_plt.patches]
        print(bottom_heights)
        n_annot_plt = sns.barplot(
            x=df_sample["Method"],
            y=df_sample.apply(
            lambda row: (row['n_paths'] / norm_values_n_paths[row['Name']]),
            axis=1) * (1-(df_sample["n_paths"]-df_sample["n_annotations"])/df_sample["n_paths"]),
            edgecolor='black',
            ax=ax,
            palette=self.palette1_bf,
            label="Annotation Paths",
            bottom= bottom_heights,
            )


        ax.set_xlabel("Sample", fontsize=self.font_size)
        ax.set_ylabel("Number of Paths", fontsize=self.font_size)
        methods_with_linebreaks = [method.replace("_", "\n") for method in df_sample["Method"].unique()]
        ax.set_xticklabels(methods_with_linebreaks, rotation=0, ha='center', fontsize=self.x_tick_size)
        plt.yticks(fontsize=self.x_tick_size)
        #ax.legend(loc='upper right')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        plt.savefig(f"{self.out_path}/plots/paths_annotation_stacked_relative_plt.svg")
        self.plot_dict["paths_annotation_stacked_relative_plt"] = f"../plots/paths_annotation_stacked_relative_plt.svg"
        # save as png
        plt.savefig(f"{self.out_path}/plots/paths_annotation_stacked_relative_plt.png")
        #plt.show()

        return fig, ax
    
    def attempts_paths_plt(
            self,
            df_sample: pd.DataFrame = None,
    ):
        """
        This plot shows the number of attempts per number of paths(of the original
        bioformats file). Each Method is its own line.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        #noise = np.random.normal(-0.1, 0.1, size=len(df_sample))
        df = df_sample[df_sample["Method"]!="Bioformats"]
        noise = np.random.normal(-0.1, 0.1, size=len(df))
        plot = sns.scatterplot(
            x=df["og_n_paths"],
            y=df["attempts"]+noise,
            hue=df["Method"],
            ax=ax,
            palette=self.palette0,
            alpha=0.9,
            s=self.scatter_size)
        
        ax.set_xlabel("Number of Paths", fontsize=self.font_size)
        ax.set_ylabel("Number of Attempts", fontsize=self.font_size)
        plt.yticks(fontsize=self.x_tick_size)
        plt.xticks(fontsize=self.x_tick_size)
        ax.legend(loc='upper left')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        plt.savefig(f"{self.out_path}/plots/attempts_paths_plt.svg")
        self.plot_dict["attempts_paths_plt"] = f"../plots/attempts_paths_plt.svg"
        # save as png
        plt.savefig(f"{self.out_path}/plots/attempts_paths_plt.png")
        #plt.show()

    def method_edit_distance_no_annot_plt(
            self,
            df_sample: pd.DataFrame = None,
    ):
        """
        This function creates a plot which compares th
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        df = df_sample[df_sample["Method"]!="Bioformats"]
        plot = sns.barplot(
            x=df["Method"],
            y=df["Edit_distance_no_annot"],
            edgecolor='black',
            ax=ax,
            palette=self.palette0)

        ax.set_xlabel("Method", fontsize=self.font_size)
        ax.set_ylabel("Edit Distance", fontsize=self.font_size)
        methods_with_linebreaks = [method.replace("_", "\n") for method in df["Method"].unique()]
        ax.set_xticklabels(methods_with_linebreaks, rotation=0, ha='center', fontsize=self.x_tick_size)
        plt.yticks(fontsize=self.x_tick_size)

        #ax.legend(loc='upper right')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        plt.savefig(f"{self.out_path}/plots/method_edit_distance_no_annot_plt.svg")
        self.plot_dict["method_edit_distance_no_annot_plt"] = f"../plots/method_edit_distance_no_annot_plt.svg"
        # save as png
        plt.savefig(f"{self.out_path}/plots/method_edit_distance_no_annot_plt.png")
        #plt.show()

        return fig, ax
    
    def method_edit_distance_only_annot_plt(
            self,
            df_sample: pd.DataFrame = None,
    ):
        """
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        df = df_sample[df_sample["Method"]!="Bioformats"]
        plot = sns.barplot(
            x=df["Method"],
            y=df["Edit_distance_only_annot"],
            edgecolor='black',
            ax=ax,
            palette=self.palette0)

        ax.set_xlabel("Method", fontsize=self.font_size)
        ax.set_ylabel("Edit Distance", fontsize=self.font_size)
        methods_with_linebreaks = [method.replace("_", "\n") for method in df["Method"].unique()]
        ax.set_xticklabels(methods_with_linebreaks, rotation=0, ha='center', fontsize=self.x_tick_size)
        plt.yticks(fontsize=self.x_tick_size)
        #ax.legend(loc='upper right')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        plt.savefig(f"{self.out_path}/plots/method_edit_distance_only_annot_plt.svg")
        self.plot_dict["method_edit_distance_only_annot_plt"] = f"../plots/method_edit_distance_only_annot_plt.svg"
        # save as png
        plt.savefig(f"{self.out_path}/plots/method_edit_distance_only_annot_plt.png")
        #plt.show()

        return fig, ax
    
    def n_paths_cost_plt(
            self,
            df_sample: pd.DataFrame = None,
    ):
        """
        This function creates a plot which compares the number of ground truth
        paths with the cost of the prediction for that gt. Each data point is a
        dot and the color represents the method.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        df = df_sample[df_sample["Method"]!="Bioformats"]
        plot = sns.scatterplot(
            x=df["og_n_paths"],
            y=df["cost"],
            hue=df["Method"],
            ax=ax,
            palette=self.palette0,
            s=self.scatter_size)
        
        ax.set_xlabel("Number of Paths", fontsize=self.font_size)
        ax.set_ylabel("Cost in $", fontsize=self.font_size)
        ax.legend(loc='upper right')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.yticks(fontsize=self.x_tick_size)
        plt.xticks(fontsize=self.x_tick_size)
        plt.tight_layout()

        plt.savefig(f"{self.out_path}/plots/n_paths_cost_plt.svg")
        self.plot_dict["n_paths_cost_plt"] = f"../plots/n_paths_cost_plt.svg"
        # save as png
        plt.savefig(f"{self.out_path}/plots/n_paths_cost_plt.png")
        #plt.show()

        return fig, ax
    def  n_paths_time_plt(
            self,
            df_sample: pd.DataFrame = None,
    ):
        """
        This function creates a plot which compares the number of ground truth
        paths with the time it took to generate the prediction. Each data point is a
        dot and the color represents the method.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        df = df_sample[df_sample["Method"]!="Bioformats"]
        plot = sns.scatterplot(
            x=df["og_n_paths"],
            y=df["time"],
            hue=df_sample["Method"],
            ax=ax,
            palette=self.palette0,
            s=self.scatter_size)
        
        ax.set_xlabel("Number of Paths", fontsize=self.font_size)
        ax.set_ylabel("Time in s", fontsize=self.font_size)
        
        ax.legend(loc='upper right')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.yticks(fontsize=self.x_tick_size)
        plt.xticks(fontsize=self.x_tick_size)
        plt.tight_layout()

        plt.savefig(f"{self.out_path}/plots/n_paths_time_plt.svg")
        self.plot_dict["n_paths_time_plt"] = f"../plots/n_paths_time_plt.svg"
        # save as png
        plt.savefig(f"{self.out_path}/plots/n_paths_time_plt.png")
        #plt.show()

        return fig, ax
    
    
    
    
    def plot_price_per_token(self):
        class PriceDevelopments(BaseModel):
            model: str
            price_in: float # the price per million tokens
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

        # create a dataframe
        import seaborn.objects as so
        df = pd.DataFrame([m.dict() for m in dataset])
        df["price_per_mmlu"] = df["price_in"]/df["mmlu"]
        df["model_family"] = df["model"].str.split("-").str[0]

        fig,ax = plt.subplots(figsize=self.figsize)
        reference_date = datetime.date(1970, 1, 1)

        # Convert the 'release_date' column to the number of days since the reference date
        df['release_days'] = (df['release_date'] - reference_date).apply(lambda x: x.days)

        sns.regplot(data=df, x="release_days", y="price_per_mmlu", order=1, color=".3")
        sns.scatterplot(data=df, x="release_date", y="price_per_mmlu", hue="model_family", s=self.scatter_size)

        plt.ylabel("Price per MMLU", fontsize=self.font_size)
        plt.xlabel("Release Date", fontsize=self.font_size)
        plt.yticks(fontsize=self.x_tick_size)
        plt.xticks(fontsize=self.x_tick_size)
        plt.xticks(rotation=45)
        # grid
        ax.legend(loc='upper right')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)        # save as svg
        plt.savefig(f"{self.out_path}/plots/plot_price_per_token.svg")
        # save as png
        plt.savefig(f"{self.out_path}/plots/plot_price_per_token.png")
        self.plot_dict["plot_price_per_token"] = f"../plots/plot_price_per_token.svg"

        plt.show()


# Which plots do I want to return?
# plot which shows deviation between runs of same sample
# plot which shows the datatype dependent performance
# plot which shows the method dependent performance
# plot which shows the cost to run the tool 
# maybe show the average per sample std instead of the std of the entire dataset
# maybe look at tempertature dependent performance