import time
import xml.etree.ElementTree as ET
from typing import Union, Any
import numpy as np
from zss import simple_distance, Node
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from DataClasses import Sample
from DataClasses import Experiment
import tree
from PyGram import Profile
from deprecated import deprecated


class OMEEvaluator:
    """
    This class evaluates the performance of a OME XML generation model by calculating the edit distance between the
    ground truth and the prediction. https://github.com/timtadh/zhang-shasha
    """

    def __init__(self,
                 schema: str = None,
                 experiment: Experiment = None,
                 out_path: str = None):
        """
        :param path_to_raw_metadata: path to the raw metadata file
        """
        self.all_paths = None
        self.gt_graph = None
        self.predicted = [self.string_to_ome_xml(x.metadata_str) for x in experiment.samples.values()]
        self.pred_graph = None
        self.out_path: str = out_path
        self.edit_score = []
        self.experiment = experiment
        self.plot_dict = {}
        for s in self.experiment.samples.values():
            s.metadata_xml = self.string_to_ome_xml(s.metadata_str)
            s.paths = self.get_paths(s.metadata_xml)

        self.report()
        print("test")

    ####################################################################################################################
    def element_to_pygram(self, element: ET.Element):
        """
        Convert an xml element to a pygram tree.
        """
        node = tree.Node(element.tag)
        for child in element:
            node.addkid(self.element_to_pygram(child))
        return node
    
    ####################################################################################################################
    def zss_edit_distance(self, xml_a: ET.Element, xml_b: ET.Element):
        """
        Calculate the edit distance between two xml trees on word level.
        Here an outline of the algorithm:
        1. Get the paths of the xml trees.
        2. Align the paths such that the distance between the paths is minimized.
        3. Calculate the word level edit distance between the paths.
        """
        self.pred_graph = self.get_graph(xml_a)
        self.gt_graph = self.get_graph(xml_b)
        return simple_distance(self.gt_graph, self.pred_graph)
    
    ####################################################################################################################
    def pygram_edit_distance(self, xml_a: ET.Element, xml_b: ET.Element):
        """
        Calculate the edit distance between two xml trees on word level.
        Here an outline of the algorithm:
        """
        profile1 = Profile(self.element_to_pygram(xml_a), 2, 3)
        profile2 = Profile(self.element_to_pygram(xml_b), 2, 3)
        return profile1.edit_distance(profile2)
    
    ####################################################################################################################
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
    
    ####################################################################################################################
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
    
    ####################################################################################################################
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
    
    ####################################################################################################################
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
        print(dp)
        return dp[-1][-1], alignment_a, alignment_b
    
    ####################################################################################################################
    def get_paths(self, xml_root: ET.Element, path: str = '') -> set:
        """
        Helper function to get all paths in an XML tree.
        :return: set of paths
        """
        paths = set()
        for child in xml_root:
            new_path = path + '/' + child.tag.split('}')[1]
            if child.attrib:
                for key in child.attrib.keys():
                    paths.add(new_path + '/' + key + '=' + child.attrib[key])
                    paths.update(self.get_paths(child, new_path))
            else:
                paths.add(new_path)
                paths.update(self.get_paths(child, new_path))
        return paths
    
    ####################################################################################################################
    def get_graph(self, xml_root: ET.Element, root=None):
        """
        Helper function to get the graph representation of an XML tree.
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
    
    ####################################################################################################################
    def path_difference(self, xml_a: ET.Element, xml_b: ET.Element):
        """
        Calculates the length of the difference between the path sets in two xml trees.
        """
        paths_a = self.get_paths(xml_a)
        paths_b = self.get_paths(xml_b)
        print("path difference: ", paths_a)
        return len(paths_a.symmetric_difference(paths_b))
    
    #################################################################################################################### 
    def read_ome_xml(self, path: str):
        """
        This method reads the ome xml file and returns the root element.
        """
        tree = ET.parse(path)
        root = tree.getroot()
        print("root", tree)
        return root

    def string_to_ome_xml(self, string):
        """
        This method reads the ome xml string and returns the root element.
        """
        root = ET.fromstring(string)
        return root

    def flatten(self, container):
        for i in container:
            if isinstance(i, (list, tuple, set)):
                yield from self.flatten(i)
            else:
                yield i
    
    ####################################################################################################################
    def report(self):
        """
        Write evaluation report to file.
        """
        with open(f"{self.out_path}/reports/report_test.md", "w") as f:
            f.write("# Evaluation Report\n")
            f.write("## File content\n")
            f.write(f"### File Ground Truth: \n")
            # create a dataframe with the paths
            df_paths = self.path_df()
            print(df_paths)
            # create a dataframe with the samples properties
            df_sample = self.sample_df(df_paths)
            print(df_sample)
            # create the plots
            self.method_edit_distance_plt(df_sample)
            self.n_paths_method_plt(df_sample)
            self.format_method_plot(df_sample)
            # add the plots to the report
            f.write("## Path Comparison\n")
            for k, v in self.plot_dict.items():
                f.write(f"![{k}]({v})\n")

    ####################################################################################################################
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
        df["Method"] = [s.method if s.method else None for s in self.experiment.samples.values()]
        df["Name"] = [s.name if s.name else None for s in self.experiment.samples.values()]
        df["n_paths"] = df_paths.sum()
        df["n_annotations"] = {k: df_paths[k][df_paths.index.str.contains("StructuredAnnotations")].sum() for k in
                               df_paths.columns}
        df["og_image_format"] = [s.format if s.format else None for s in self.experiment.samples.values()]
        edit_distances = []

        for n in df["Name"].unique():
            methods = list(df["Method"].unique())
            for m in methods:
                gt = self.experiment.samples[f"{n}_Bioformats"].metadata_xml
                test = self.experiment.samples[f"{n}_{m}"].metadata_xml
                t0 = time.time()
                #edit_distances.append(self.zss_edit_distance(test, gt))
                edit_distances.append(self.pygram_edit_distance(test, gt))
                t1 = time.time()
        df["Edit_distance"] = edit_distances
        return df

    ####################################################################################################################
    def path_df(
            self,
    ):
        """
        TODO: Add docstring
        """
        self.all_paths = pd.Series(list(set(self.flatten([s.paths for s in self.experiment.samples.values()]))))
        df = pd.DataFrame(columns=[f"{s.name}_{s.method}" for s in self.experiment.samples.values()],
                          index=self.all_paths)
        for name, method, path in zip([s.name for s in self.experiment.samples.values()],
                                      [s.method for s in self.experiment.samples.values()],
                                      [s.paths for s in self.experiment.samples.values()]):
            df[f"{name}_{method}"] = df.index.isin(path)
        return df

    ####################################################################################################################
    def method_edit_distance_plt(
            self,
            df_sample: pd.DataFrame = None,
    ):
        """
        This function creates a plot which compares the inter sample standard deviation.
        The X-axis will be the used method, whereas the Y-axis will be the standard deviation.
        """
        fig, ax = plt.subplots()
        sns.barplot(x="Method", y="Edit_distance", data=df_sample, ax=ax)
        plt.savefig(f"{self.out_path}/plots/method_edit_distance_plt.svg")
        self.plot_dict["method_edit_distance_plt"] = f"../plots/method_edit_distance_plt.svg"
        return fig, ax

    ####################################################################################################################
    import matplotlib.pyplot as plt

    def n_paths_method_plt(self, df_sample: pd.DataFrame = None):
        """
        Plots the number of paths per method as a bar plot.
        
        Parameters:
        - df_sample: pd.DataFrame, a DataFrame containing the data to plot.
        
        Returns:
        - fig: The figure object.
        - ax: The axes object.
        """
        # Create the figure and axis objects
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create the bar plot
        ax.bar(df_sample["Method"], df_sample["n_paths"], color="skyblue", edgecolor='black')

        # Add labels and title
        ax.set_xlabel("Method", fontsize=14)
        ax.set_ylabel("Number of Paths", fontsize=14)
        ax.set_title("Number of Paths per Method", fontsize=16)
        ax.tick_params(axis='x', rotation=45)

        # Customize the appearance
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Save the plot
        plt.tight_layout()
        plt.savefig(f"{self.out_path}/plots/n_paths_method_plt.svg")

        # Update the plot dictionary
        self.plot_dict["n_paths_method_plt"] = f"../plots/n_paths_method_plt.svg"

        return fig, ax


    ####################################################################################################################
    def format_method_plot(
            self,
            df_sample: pd.DataFrame = None,
    ):
        """
        This plot compares the performance of the different methods based on the original image format.
        For each method several bars are plotted, one for each image format.
        """
        fig, ax = plt.subplots()
        sns.barplot(x="Method", y="Edit_distance", hue="og_image_format", data=df_sample, ax=ax)
        plt.savefig(f"{self.out_path}/plots/format_method_plot.svg")
        self.plot_dict["format_method_plot"] = f"../plots/format_method_plot.svg"
        plt.show()
        return fig, ax

# Which plots do I want to return?
# plot which shows deviation between runs of same sample
# plot which shows the datatype dependent performance
# plot which shows the method dependent performance
# plot which shows the cost to run the tool 
# maybe show the average per sample std instead of the std of the entire dataset