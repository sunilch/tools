
from pandas.core.api import Series
from pandas.core.categorical import Categorical
import pandas.core.algorithms as algos
import pandas.core.common as com
import pandas.core.nanops as nanops
import pandas.lib as lib
from pandas.compat import zip
import numpy as np

def qcutnew(x, q, labels=None, retbins=False, precision=3):
    """
    Quantile-based discretization function. Discretize variable into
    equal-sized buckets based on rank or based on sample quantiles. For example
    1000 values for 10 quantiles would produce a Categorical object indicating
    quantile membership for each data point.
    Parameters
    ----------
    x : ndarray or Series
    q : integer or array of quantiles
        Number of quantiles. 10 for deciles, 4 for quartiles, etc. Alternately
        array of quantiles, e.g. [0, .25, .5, .75, 1.] for quartiles
    labels : array or boolean, default None
        Used as labels for the resulting bins. Must be of the same length as
        the resulting bins. If False, return only integer indicators of the
        bins.
    retbins : bool, optional
        Whether to return the bins or not. Can be useful if bins is given
        as a scalar.
    precision : int
        The precision at which to store and display the bins labels
    Returns
    -------
    out : Categorical or Series or array of integers if labels is False
        The return type (Categorical or Series) depends on the input: a Series
        of type category if input is a Series else Categorical. Bins are
        represented as categories when categorical data is returned.
    bins : ndarray of floats
        Returned only if `retbins` is True.
    Notes
    -----
    Out of bounds values will be NA in the resulting Categorical object
    """
    if com.is_integer(q):
        quantiles = np.linspace(0, 1, q + 1)
    else:
        quantiles = q
    bins = algos.quantile(x, quantiles)
    return _bins_to_cuts_new(x, bins, labels=labels, retbins=retbins,
                         precision=precision, include_lowest=True)

def _bins_to_cuts_new(x, bins, right=True, labels=None, retbins=False,
                  precision=3, name=None, include_lowest=False):
    x_is_series = isinstance(x, Series)
    series_index = None

    #Added this line to the original code
    bins =np.array(sorted(list(set(bins))))

    if x_is_series:
        series_index = x.index
        if name is None:
            name = x.name

    x = np.asarray(x)

    side = 'left' if right else 'right'
    ids = bins.searchsorted(x, side=side)

    if len(algos.unique(bins)) < len(bins):
        raise ValueError('Bin edges must be unique: %s' % repr(bins))

    if include_lowest:
        ids[x == bins[0]] = 1

    na_mask = com.isnull(x) | (ids == len(bins)) | (ids == 0)
    has_nas = na_mask.any()

    if labels is not False:
        if labels is None:
            increases = 0
            while True:
                try:
                    levels = _format_levels(bins, precision, right=right,
                                            include_lowest=include_lowest)
                except ValueError:
                    increases += 1
                    precision += 1
                    if increases >= 20:
                        raise
                else:
                    break

        else:
            if len(labels) != len(bins) - 1:
                raise ValueError('Bin labels must be one fewer than '
                                 'the number of bin edges')
            levels = labels

        levels = np.asarray(levels, dtype=object)
        np.putmask(ids, na_mask, 0)
        fac = Categorical(ids - 1, levels, ordered=True, fastpath=True)
    else:
        fac = ids - 1
        if has_nas:
            fac = fac.astype(np.float64)
            np.putmask(fac, na_mask, np.nan)

    if x_is_series:
        fac = Series(fac, index=series_index, name=name)

    if not retbins:
        return fac

    return fac, bins


def _format_levels(bins, prec, right=True,
                   include_lowest=False):
    fmt = lambda v: _format_label(v, precision=prec)
    if right:
        levels = []
        for a, b in zip(bins, bins[1:]):
            fa, fb = fmt(a), fmt(b)

            if a != b and fa == fb:
                raise ValueError('precision too low')

            formatted = '(%s, %s]' % (fa, fb)

            levels.append(formatted)

        if include_lowest:
            levels[0] = '[' + levels[0][1:]
    else:
        levels = ['[%s, %s)' % (fmt(a), fmt(b))
                  for a, b in zip(bins, bins[1:])]

    return levels


def _format_label(x, precision=3):
    fmt_str = '%%.%dg' % precision
    if np.isinf(x):
        return str(x)
    elif com.is_float(x):
        frac, whole = np.modf(x)
        sgn = '-' if x < 0 else ''
        whole = abs(whole)
        if frac != 0.0:
            val = fmt_str % frac

            # rounded up or down
            if '.' not in val:
                if x < 0:
                    return '%d' % (-whole - 1)
                else:
                    return '%d' % (whole + 1)

            if 'e' in val:
                return _trim_zeros(fmt_str % x)
            else:
                val = _trim_zeros(val)
                if '.' in val:
                    return sgn + '.'.join(('%d' % whole, val.split('.')[1]))
                else:  # pragma: no cover
                    return sgn + '.'.join(('%d' % whole, val))
        else:
            return sgn + '%0.f' % whole
    else:
        return str(x)


def _trim_zeros(x):
    while len(x) > 1 and x[-1] == '0':
        x = x[:-1]
    if len(x) > 1 and x[-1] == '.':
        x = x[:-1]
    return x

if __name__=='__main__':
    print qcutnew(range(25),3,retbins=True)

    print 'now testing the corner case',[0]*20+range(20)
    print qcutnew([0]*20+range(20),10)
