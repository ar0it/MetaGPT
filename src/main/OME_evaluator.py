import xml.etree.ElementTree as ET
import javabridge
import bioformats


# javabridge.start_vm(class_path=bioformats.JARS)


class OMEEvaluator:
    """
    This class evaluates the performance of a OME XML generation model by calculating the similarity between the generated
    and the ground truth OME XML. The similarity is defined as the number of identical nodes divided by the total number of
    nodes in the ground truth OME XML.
    """

    def __init__(self, path_to_raw_metadata=None,
                 gt_path="/home/aaron/PycharmProjects/MetaGPT/raw_data/testetst_Image8_edited_.ome.xml",
                 pred_path="/home/aaron/PycharmProjects/MetaGPT/out/ome_xml.ome.xml"):
        """
        :param path_to_raw_metadata: path to the raw metadata file
        """

        self.prediction = self.read_ome_xml(pred_path)
        self.ground_truth = self.read_ome_xml(gt_path)

        print(self.edit_distance(xml_a=self.prediction, xml_b=self.ground_truth))
        # Example usage

    def edit_distance(self, xml_a, xml_b):
        """
        Calculate the edit distance between two xml trees on word level.
        Here an outline of the algorithm:
        1. Get the paths of the xml trees.
        2. Align the paths such that the distance between the paths is minimized.
        3. Calculate the word level edit distance between the paths.
        """
        # get the paths of the xml trees

        xml_a_paths = [[[y] for y in x.split("/") if y] for x in self.get_paths(xml_a)]
        xml_b_paths = [[[y] for y in x.split("/") if y] for x in self.get_paths(xml_b)]
        print("xml_a_paths", xml_a_paths)
        print("xml_a_paths", xml_b_paths)

        # align the paths
        s1 = [[["a"], ["a"], ["a"]], [["b"], ["b"]], [["c"]], [["d"]]]
        s2 = [[["a"], ["a"], ["a"]], [["b"], ["b"]], [["d"]], [["c"]], [["d"]]]
        score, alignment_a, alignment_b = self.align_paths(xml_a_paths, xml_b_paths)
        print("Score:", score)
        print("Alignment 1:", alignment_a)
        print("Alignment 2:", alignment_b)

    def align_paths(self, paths_a, paths_b):
        """
        Align the paths such that the sum of distances between the paths is minimized.
        paths_a: set of paths
        paths_b: set of paths
        :return: list of tuples of aligned paths
        """
        score, alignment_a, alignment_b = self.align_sequences(paths_a, paths_b, cost=self.align_sequences_score)
        return score, alignment_a, alignment_b

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
        m, n = len(s1), len(s2)
        # Create a matrix to store the cost of edits
        dp = [[0] * (n + 1) for _ in range(m + 1)]

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
            alignment_b = [[["-"]*len(s1[j - 1])]] + alignment_b
            i -= 1
        while j > 0:
            alignment_a = [[["-"]*len(s2[j - 1])]] + alignment_a
            alignment_b = [s2[j - 1]] + alignment_b
            j -= 1

        return dp[-1][-1], alignment_a, alignment_b

    def word_edit_distance(self, aligned_paths):
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

    def get_paths(self, xml_root, path=''):
        """
        Helper function to get all paths in an XML tree.
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
        return root

    def read_ome_tiff(self, path):
        """
        This method reads the ome tiff file.
        """
        ome = bioformats.OMEXML()
        print(ome)
        return ome

    def evaluate(self):
        """
        compare the two ome xml trees and return the similarity score by calculating the edit distance between the two xml.
        """
        print("Evaluation in progress...")
        self.score = self.path_difference(self.prediction, self.ground_truth)

    def report(self):
        """
        Write evaluation report to file.
        """
        with open(f"out/report_{self.path_ground_truth.split('/')[-1]}.txt", "w") as f:
            f.write("Evaluation Report\n")
            f.write(f"File: {self.path_ground_truth.split('/')[-1]}\n")
            f.write(f"Score: {self.score}\n")


# javabridge.kill_vm()

if __name__ == "__main__":
    evaluator = OMEEvaluator()
