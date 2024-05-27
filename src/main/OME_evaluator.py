import xml.etree.ElementTree as ET
#import javabridge
#import bioformats
import numpy as np
from zss import simple_distance, Node
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# javabridge.start_vm(class_path=bioformats.JARS)


class OMEEvaluator:
    """
    This class evaluates the performance of a OME XML generation model by calculating the edit distance between the
    ground truth and the prediction. https://github.com/timtadh/zhang-shasha
    """

    def __init__(self,
                 schema: str = None,
                 ground_truth: str = None,
                 predicted: list[str] = None,
                 out_path: str = None):
        """
        :param path_to_raw_metadata: path to the raw metadata file
        """
        self.gt_graph = None
        self.predicted = [self.string_to_ome_xml(x) for x in predicted]
        self.pred_graph = None
        self.ground_truth = self.string_to_ome_xml(ground_truth)
        self.out_path: str = out_path
        self.edit_score = []
        self.samples = {}
        for p in self.predicted:
            self.edit_score.append(self.edit_distance(p, self.ground_truth))
        self.report()

    def edit_distance(self, xml_a:ET.Element, xml_b:ET.Element):
        """
        Calculate the edit distance between two xml trees on word level.
        Here an outline of the algorithm:
        1. Get the paths of the xml trees.
        2. Align the paths such that the distance between the paths is minimized.
        3. Calculate the word level edit distance between the paths.
        """
        # get the paths of the xml trees

        self.pred_graph = self.get_graph(xml_a)
        self.gt_graph = self.get_graph(xml_b)
        return simple_distance(self.gt_graph, self.pred_graph)

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

    def get_paths(self, xml_root:ET.Element, path: str = '') -> set:
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

    def get_graph(self, xml_root:ET.Element, root=None):
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

    def path_difference(self, xml_a, xml_b):
        """
        Calculates the length of the difference between the path sets in two xml trees.
        """
        paths_a = self.get_paths(xml_a)
        paths_b = self.get_paths(xml_b)
        print("path difference: ", paths_a)
        return len(paths_a.symmetric_difference(paths_b))

    def read_ome_xml(self, path):
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
    def read_ome_tiff(self, path):
        """
        This method reads the ome tiff file.
        """
        pass
        #ome = bioformats.OMEXML()
        #print(ome)
        #return ome

    def evaluate(self):
        """
        compare the two ome xml trees and return the similarity score by calculating the edit distance between the two xml.
        """
        print("Evaluation in progress...")
        self.score = self.path_difference(self.prediction, self.ground_truth)

    def flatten(self, container):
        for i in container:
            if isinstance(i, (list, tuple, set)):
                yield from self.flatten(i)
            else:
                yield i

    def report(self):
        """
        Write evaluation report to file.
        """
        with open(f"../../out/report_test.md", "w") as f:
            f.write("# Evaluation Report\n")
            f.write("## File content\n")
            f.write(f"### File Ground Truth: \n")
            paths_gt = self.get_paths(self.ground_truth)
            self.samples["GT"] = paths_gt
            for j, p in enumerate(self.predicted):
                f.write(f"### Prediction {j}\n")
                self.samples[f"Pred{j}"] = self.get_paths(p)
            plt.savefig(f"{self.out_path}.png")
            # add the plot to the report
            f.write("## Path Comparison\n")
            f.write(f"![barplot comparing the xml files]({self.out_path}.png)\n")

    def sample_df(
            self,
            df_paths: pd.DataFrame = None,
    ):
        """
        This function creates a df with samples as Index and properties as Columns
        TODO: Autogenerated function controll if it works as expected
        """
        properties = ["Method", "n_paths", "n_annotations", "Edit_distance"]
        df = pd.DataFrame(index=df_paths.columns, columns=properties)
        df["Method"] = "qwe"
        df["Name"] = "qwe"
        df["iter"] = 1
        df["n_paths"] = df_paths.sum()
        df["n_annotations"] = df_paths[df_paths.__contains__("StructuredAnnotations")].sum()
        df["Edit_distance"] = [self.edit_distance(x, self.ground_truth) for x in self.prediction]
        return df

    def path_df(
            self,
            paths: dict[list[str]] = None  # each sample in the dictonary has a list of paths
    ):
        df = pd.DataFrame()
        for sample in paths:
            for path in paths[sample]:
                df.at[sample, path] = 1
        return df

    def sample_std_plot(
            self,
            df_sample: pd.DataFrame = None,
    ):
        """
        This function creates a plot which compares the inter sample standard deviation.
        The X-axis will be the used method, whereas the Y-axis will be the standard deviation.
        """
        df_sample["Sample_std"] = df_sample.groupby(['Method', 'Name'])['n_paths'].std().reset_index()
        plt.bar(df_sample["Method"], df_sample["Sample_std"])


# Which plots do I want to return?
# plot which shows deviation between runs of same sample
# plot which shows the datatype dependent performance
# plot which shows the method dependent performance
# plot which shows the cost to run the tool
# maybe show the average per sample std instead of the std of the entire dataset

# javabridge.kill_vm()

if __name__ == "__main__":

    evaluator = OMEEvaluator(predicted=["<OME></OME>"], ground_truth="<OME></OME>", out_path="./out")
