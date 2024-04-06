from past.utils import old_div
import numpy


def generate_latin_hypercube_points(num_points, domain_bounds):
    """Compute a set of random points inside some domain that lie in a latin hypercube.

    In 2D, a latin hypercube is a latin square--a checkerboard--such that there is exactly one sample in
    each row and each column.  This notion is generalized for higher dimensions where each dimensional
    'slice' has precisely one sample.

    See wikipedia: http://en.wikipedia.org/wiki/Latin_hypercube_sampling
    for more details on the latin hypercube sampling process.

    :param num_points: number of random points to generate
    :type num_points: int > 0
    :param domain_bounds: [min, max] boundaries of the hypercube in each dimension
    :type domain_bounds: list of dim ClosedInterval
    :return: uniformly distributed random points inside the specified hypercube
    :rtype: array of float64 with shape (num_points, dim)

    """
    # TODO(GH-56): Allow users to pass in a random source.
    if num_points == 0:
        return numpy.array([])

    points = numpy.zeros((num_points, len(domain_bounds)), dtype=numpy.float64)
    for i, interval in enumerate(domain_bounds):
        # Cut the range into num_points slices
        subcube_edge_length = old_div(len(interval), float(num_points))

        # Create random ordering for slices
        ordering = numpy.arange(num_points)
        numpy.random.shuffle(ordering)

        for j in range(num_points):
            point_base = min(interval) + subcube_edge_length * ordering[j]
            points[j, i] = point_base + numpy.random.uniform(0.0, subcube_edge_length)

    return points