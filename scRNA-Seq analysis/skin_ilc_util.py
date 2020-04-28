import matplotlib.colors
import numpy as np
import scanpy as sc
import os
import matplotlib.pyplot as plt
import seaborn as sns

from colorspacious import cspace_converter
from matplotlib.colors import to_rgb
rgb2lab = cspace_converter("sRGB1", "CAM02-UCS")
lab2rgb = cspace_converter("CAM02-UCS", "sRGB1")

##############
# Color maps # 
##############

class LinearSegmentedColormap0(matplotlib.colors.LinearSegmentedColormap):
    def __init__(self, name, segmentdata, N=256, gamma=1.0, start_col = None):
        # self.cm = matplotlib.colors.LinearSegmentedColormap(name, segmentdata, N, gamma)
        super().__init__(name, segmentdata, N, gamma)
        # print(self.cm)
        self.start_col = matplotlib.colors.to_rgba(start_col) if start_col is not None else None

    # def __call__(self, X, alpha = None, bytes = False):
        # # return self.cm(X, alpha, bytes)
        # return super().__call__(X, alpha, bytes)

    def __call__(self, X, alpha=None, bytes=False):
        ret = super().__call__(X, alpha, bytes)
        idx = X == 0 
        print(ret)
        if self.start_col is not None:
            if np.array(X).shape == () and idx:
                    ret = self.start_col
            elif np.any(idx):
                ret[idx,:] = self.start_col
        return ret

def linear_colors(l, start_col = None):
    reds = []
    greens = []
    blues = []
    l = [matplotlib.colors.to_rgb(x) for x in l]
    xs = np.linspace(0, 1, len(l))
    for (x, elem) in zip(xs, l):
        reds.append((x, elem[0], elem[0]))
        greens.append((x, elem[1], elem[1]))
        blues.append((x, elem[2], elem[2]))
    cdict = {'red' : tuple(reds), 'green' : tuple(greens), 'blue' : tuple(blues)}
    # print(return_dict)
    if start_col is not None:
        return LinearSegmentedColormap0("Name", cdict, start_col = start_col)
    else:
        return matplotlib.colors.LinearSegmentedColormap("Name", cdict)

def linearize_palette(palette):
    palette_new = []
    #l_start = 75
    #l_end = 40
    l_start = rgb2lab(palette[0][:3])[0]
    l_end = rgb2lab(palette[-1][:3])[0]
    xs = np.linspace(0, 1, len(palette))
    for (c, x) in zip(palette, xs):
        lab = rgb2lab(c[:3])
        palette_new.append(np.clip((*lab2rgb((np.interp(x, [0, 1], [l_start, l_end]), lab[1], lab[2])), 1), 0, 1))
    return palette_new

def linearize_cmap(cmap):
    return matplotlib.colors.ListedColormap(linearize_palette(cmap(np.linspace(0, 1, 256))))

def DiscreteColormap(colors):
    cdict = {'red': [],
             'green': [],
             'blue': [],
             'alpha': []
            }
    xs = np.linspace(0, 1, len(colors) + 1)
    r, g, b, a = colors[0]
    cdict['red'].append((0, r, r))
    cdict['green'].append((0, g, g))
    cdict['blue'].append((0, b, b))
    cdict['alpha'].append((0, a, a))
    for i in range(len(colors) - 1):
        r, g, b, a = colors[i]
        r2, g2, b2, a2 = colors[i + 1]
        cdict['red'].append((xs[i+1], r, r2))
        cdict['green'].append((xs[i+1], g, g2))
        cdict['blue'].append((xs[i+1], b, b2))
        cdict['alpha'].append((xs[i+1], a, a2))
    r, g, b, a = colors[-1]
    cdict['red'].append((1, r, r))
    cdict['green'].append((1, g, g))
    cdict['blue'].append((1, b, b))
    cdict['alpha'].append((1, a, a))
    return matplotlib.colors.LinearSegmentedColormap('DiscreteCmap', cdict)

def ScaledColormap(cmap, break_in = 0.5, break_out = 0.5, name='ScaledCmap'):
    cdict = {'red': [],
                 'green': [],
                 'blue': [],
                 'alpha': []
                }

    build_index = np.linspace(0, 1, 10)

    for x in build_index:
        y = np.interp(x, [0, break_in, 1], [0, break_out, 1])
        #print(x, y)
        r, g, b, a = cmap(x)
        r2, g2, b2, a2 = cmap(y)
        lab = rgb2lab((r, g, b))
        lab2 = rgb2lab((r2, g2, b2))
        r, g, b = lab2rgb((lab2[0], lab[1], lab[2]))
        #print(r, g, b, a)

        cdict['red'].append((x, r, r))
        cdict['green'].append((x, g, g))
        cdict['blue'].append((x, b, b))
        cdict['alpha'].append((x, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)

    return newcmap

class CenterNormalize(matplotlib.colors.Normalize):
    """
    Normalization for DC plotting
    """

    def __init__(self, vcenter = None, vmin = None, vmax = None, clip = False, method = np.median, invert = False):
        self.vcenter = vcenter
        self.vmin = vmin
        self.vmax = vmax
        self.method = method
        self.invert = invert
        #super().__init__(vcenter, vmin, vmax)
    
    def autoscale(self, A):
        super().autoscale(A)
        self.vcenter = self.method(A)
        self.vmin = A.min()
        self.vmax = A.max()
        #if self.vmax - self.vcenter > self.vcenter - self.vmin:
        #    self.invert = True
    
    def autoscale_None(self, A):
        super().autoscale_None(A)
        if self.vcenter is None:
            self.vcenter = self.method(A)
        if self.vmin is None:
            self.vmin = A.min()
        if self.vmax is None:
            self.vmax = A.max()
        #if self.vmax - self.vcenter < self.vcenter - self.vmin:
        #    self.invert = True
            
    
    def __call__(self, value, clip=None):
        """
        Map value to the interval [0, 1]. The clip argument is unused.
        """
        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)  # sets self.vmin, self.vmax if None

        if not self.vmin <= self.vcenter <= self.vmax:
            raise ValueError("vmin, vcenter, vmax must increase monotonically")
        #if self.invert:
        #    result = np.ma.masked_array(
        #        np.interp(result, [self.vmin, self.vcenter, self.vmax],
        #                  [1.0, 0.5, 0]), mask=np.ma.getmask(result))
        #else:
        if True:
            result = np.ma.masked_array(
                np.interp(result, [self.vmin, self.vcenter, self.vmax],
                          [0, 0.5, 1.]), mask=np.ma.getmask(result))
        if is_scalar:
            result = np.atleast_1d(result)[0]
        return result

class CappedNormalize(matplotlib.colors.Normalize):
    def __init__(self, vmin = None, vmax = None, bottom = None, top = None, clip = False):
        self.bottom = bottom
        self.top = top
        super().__init__(vmin, vmax, clip)
    
    def autoscale(self, A):
        super().autoscale(A)
        if self.bottom is None:
            self.bottom = self.vmin
        if self.top is None:
            self.top = self.vmax
        self.bottom = max(self.vmin, self.bottom)
        self.top = min(self.vmax, self.top)
    
    def autoscale_None(self, A):
        super().autoscale_None(A)
        if self.bottom is None:
            self.bottom = self.vmin
        if self.top is None:
            self.top = self.vmax
        self.bottom = max(self.vmin, self.bottom)
        self.top = min(self.vmax, self.top)
    
    def __call__(self, value, clip = None):
        if clip is None:
            clip = self.clip
        if clip:
            x, y = [self.bottom, self.top], [0, 1]
            return np.ma.masked_array(np.interp(value, x, y))
        else:
            return np.ma.masked_array((value - self.bottom)/(self.top - self.bottom))

def power_cmap(cmap, power = 1.0):
    x = np.linspace(0, 1, 257)
    colors = cmap(np.power(x, power))
    return linear_colors(colors)

day_palette_v8 = ['#42D384', '#43D1CD', '#CC8629', '#4362D1', '#D143AC']
day_palette = [matplotlib.colors.to_rgba(c) for c in day_palette_v8]

day_palette_light = []
#for c in day_palette_new:
for c in day_palette:
    lab = rgb2lab(c[:3])
    day_palette_light.append(np.clip((*lab2rgb((lab[0] + 10, lab[1], lab[2])), 1), 0, 1))

cm_black_white_discrete = DiscreteColormap([(0.8, 0.8, 0.8, 1), (0, 0, 0, 1)])

cm_topic_pre = linear_colors(list(map(to_rgb, ['#FAE622', '#1FA188', '#1964BB', '#1142AF'])))
cm_topic = ScaledColormap(linear_colors(linearize_palette(np.vstack([[0.85, 0.85, 0.85], cm_topic_pre(np.linspace(0, 1, 10))[:, :3]]))), break_in = 0.05, break_out = 0.15)
cm_prob = linear_colors(list(map(to_rgb, ['#F6EF37', '#007D94'])))
cm_purple_pre = linear_colors(list(map(to_rgb, ['#F2AE36', '#72008B'])))
cm_purple_scatter = ScaledColormap(linear_colors(linearize_palette(np.vstack([[0.85, 0.85, 0.85], cm_purple_pre(np.linspace(0, 1, 15))[:,:3]]))), break_in = 0.05, break_out = 0.15)
colors_purple2 = linear_colors(list(map(to_rgb, ['#FCB638', '#6B0082'])))
cm_purple = ScaledColormap(linear_colors(linearize_palette(np.vstack([[0.95, 0.95, 0.95], colors_purple2(np.linspace(0, 1, 15))[:,:3]]))), break_in = 0.15, break_out = 0.15)


gray_orange_red_pre = matplotlib.colors.LinearSegmentedColormap("GrayOrangeRed",
    {'red': ((0.0, 0.92, 0.9),
             (0.25, 1.0, 1.0),
             (1.0, 1.0, 1.0)),

    'green': ((0.0, 0.9, 0.9),
              (0.25, 210.0/256, 210.0/256),
              (1.0, 0.0, 0.0)),

    'blue':  ((0.0, 0.9, 0.9),
              (0.25, 0.3, 0.0),
              (1.0, 0.0, 0.0))
        })

cm_gray_red = linearize_cmap(gray_orange_red_pre)

hcl_colors_yellow_red2 = ("#002D81","#002985","#002488","#001F8A","#17198C","#33108E","#44048F","#520090","#5D0090","#680090","#71008F","#79008F","#81008E","#88008C","#8F008B","#960089","#9C0086","#A10084","#A70081","#AC007E","#B1007A","#B60076","#BB0071","#BF116C","#C31E66","#C82960","#CC3358","#D03B4F","#D44444","#D74D36","#DB551F","#DF5E00","#E06900","#E07400","#E17E00","#E38900","#E49500","#E7A200","#EBB12C","#FACF76")
hcl_colors_yellow_red2 = reversed(hcl_colors_yellow_red2)
cm_hcl_yellow_red2 = linear_colors(hcl_colors_yellow_red2)


####################
# Plotting helpers #
####################

def mpl_set_default_style(rcParams):
    rcParams['figure.figsize'] = (6.4,4.8)
    rcParams.update({'font.size': 20})
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams['font.family'] = 'Arial'
    rcParams['savefig.dpi'] = 300

def subplots_fixed_width(nrow = 1, ncol = 1, figsize = (6.4, 4.8),
                                left = 0.125,
                                right = 0.9,
                                bottom = 0.1,
                                top = 0.9,
                                wspace = 0.2,
                                hspace = 0.2,
                               **kwargs):
    """
    Create subplots with fixed width for each subplot.
    nrow, ncol: the number of rows and columns,
    figsize: the figsize of each single plot,
    left, right, bottom, top, wspace, hspace: the corresponding parameters to fig.subplots_adjust.
    """
    
    total_width = (ncol + (ncol - 1) * wspace) * figsize[0] / (right - left)
    total_height = (nrow + (nrow - 1) * hspace) * figsize[1] / (top - bottom)
    fig, axes = plt.subplots(nrow, ncol, figsize = (total_width, total_height), **kwargs)
    fig.subplots_adjust(left = left, right = right, bottom = bottom, top = top, wspace = wspace, hspace = hspace)
    return fig, axes

def annotate(ax, x1, x2, text, y_fact = 0.03, h_fact = 0.04, text_fact = 0.01, fontsize = 14, x_shrink = 0.00):
    """
    Add annotation to matplotlib axes object
    """

    bottom, top = ax.get_ylim()
    span = top - bottom
    y = top + span * y_fact
    h = span * h_fact
    x_span = x2 - x1
    x1 = x1 + x_shrink * x_span
    x2 = x2 - x_shrink * x_span
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c = 'k', clip_on = False)
    ax.text((x1+x2)*.5, y+h + text_fact * span, text, ha='center', va='bottom', color='k', fontsize = fontsize)


# def colors_at_time(l):
#     reds = []
#     greens = []
#     blues = []
#     for elem in l:
#         reds.append((elem[0], elem[1]/256, elem[1]/256))
#         greens.append((elem[0], elem[2]/256, elem[2]/256))
#         blues.append((elem[0], elem[3]/256, elem[3]/256))
#     return_dict = {'red' : tuple(reds), 'green' : tuple(greens), 'blue' : tuple(blues)}
#     # print(return_dict)
#     return return_dict

# # -70, -230, 60, 70, 30, 15, 90, 0.1, 0.55, 40
# # hcl_colors_yellow_red = ("#5C007F","#54007F","#4C0080","#440081","#3B0E81","#311C82","#242682","#0F2E83","#003583","#003C84","#004184","#004784","#004C85","#005185","#005685","#005A85","#005F85","#006385","#006884","#006C84","#007184","#007583","#007983","#007E82","#008282","#008781","#008C80","#009080","#00957F","#009A7E","#00A07E","#00A57D","#00AB7D","#00B17D","#00B77D","#15BF7E","#42C681","#5FCF86","#7BDB90","#C6ECC9")
# # -112, 52, 60, 70, 30, 15, 90, 0.1, 0.55, 40
# hcl_colors_yellow_red = ("#002D81","#002981","#002682","#002382","#182082","#2F1D82","#3E1B82","#491982","#531782","#5C1681","#631681","#6B1780","#711880","#771A7F","#7D1D7F","#83207E","#88237D","#8E277C","#932B7B","#97307A","#9C3479","#A13977","#A53E76","#AA4375","#AE4973","#B34E72","#B75470","#BB5A6E","#C0606D","#C4676B","#C86E6A","#CD7568","#D27C67","#D78466","#DC8C66","#E19666","#E7A06B","#EDAC71","#F6BB7D","#F9DEC2")
# hcl_colors_yellow_red = reversed(hcl_colors_yellow_red)
# hcl_yellow_red = linear_colors(hcl_colors_yellow_red)

# hcl_colors_yellow_red2 = ("#002D81","#002982","#002582","#002282","#1D1E82","#331B82","#411882","#4D1582","#571381","#5F1281","#671180","#6E127F","#74137F","#7A167E","#80197D","#861C7B","#8B207A","#902579","#952A77","#9A2E76","#9E3374","#A33972","#A73E70","#AB446E","#AF496C","#B34F69","#B75567","#BB5B64","#BF6262","#C3685F","#C76F5C","#CB775A","#D07E57","#D48755","#D98F53","#DE9952","#E4A351","#EBAF52","#F5BD53","#FFDD86")
# hcl_colors_yellow_red2 = ("#002D81","#002982","#002582","#002182","#1C1D82","#321982","#411682","#4C1281","#560F81","#5E0D80","#660B7F","#6C0B7E","#730D7D","#790F7C","#7E137A","#831679","#881B77","#8D1F76","#922474","#962972","#9A2E70","#9E336E","#A2396C","#A63E69","#AA4367","#AE4964","#B14F61","#B5555E","#B85B5B","#BC6258","#BF6855","#C36F52","#C7764E","#CB7E4B","#CF8648","#D38F46","#D99944","#DFA443","#E9B241","#FACF76")
# # -112, 61, 60, 120, 70, 15, 85, 0.5, 0.55, 40

# hcl_colors_yellow_red2 = reversed(hcl_colors_yellow_red2)
# hcl_yellow_red2 = linear_colors(hcl_colors_yellow_red2)


# 0, -270, 90, 100, 75, 15, 96, 2, 1.1, 40
# hcl_colors_expression = ("#700013","#7B002D","#850042","#8F0055","#970067","#9D0079","#A2008B","#A10099","#9E00A5","#9800B0","#9010B9","#8631C0","#7945C6","#6956CB","#5664CF","#3A70D1","#007CD3","#0087D3","#0091D3","#009AD2","#00A2D0","#00ABCE","#00B2CB","#00B9C7","#00C0C3","#00C6BE","#00CCB9","#00D2B4","#00D7AE","#00DCA8","#00E1A2","#47E59C","#68E997","#82ED91","#98F08C","#ADF388","#C0F685","#D2F884","#E3FA84","#F3FB85")
# hcl_colors_expression = reversed(hcl_colors_expression)
# hcl_expression = linear_colors(hcl_colors_expression)


# gray_orange_red = matplotlib.colors.LinearSegmentedColormap("GrayOrangeRed",
#     {'red': ((0.0, 0.8, 0.8),
#              (0.5, 1.0, 1.0),
#              (1.0, 1.0, 1.0)),

#     'green': ((0.0, 0.8, 0.8),
#               (0.5, 190.0/256, 190.0/256),
#               (1.0, 0.0, 0.0)),

#     'blue':  ((0.0, 0.8, 0.8),
#               (1.0, 0.0, 0.0))
#         })



# purple_pal = [[0.92, 0.92, 0.92]] + sns.cubehelix_palette(start = 2.5, rot = 0.6, dark = 0.25, light = 0.8, reverse = False, hue = 0.9, n_colors = 15)
# purple_pal = [[0.9, 0.9, 0.9]] + sns.cubehelix_palette(start = 2.4, rot = 0.8, dark = 0.25, light = 0.8, reverse = False, hue = 1.0, n_colors = 15)
# purple_pal = [[0.9, 0.9, 0.9]] + sns.cubehelix_palette(start = 0.0, rot = 0.6, dark = 0.35, light = 0.8, reverse = False, hue = 1.0, n_colors = 15)
#cm_purple = linear_colors(linearize_palette(purple_pal))
# cm_purple = ScaledColormap(linear_colors(linearize_palette(np.vstack([[0.95, 0.95, 0.95], cm_test(np.linspace(0, 1, 15))[:,:3]]))), break_in = 0.05, break_out = 0.15)
# cm_purple_scatter = ScaledColormap(linear_colors(linearize_palette(np.vstack([[0.85, 0.85, 0.85], cm_test(np.linspace(0, 1, 15))[:,:3]]))), break_in = 0.05, break_out = 0.15)

# colors_purple2 = linear_colors(list(map(to_rgb, ['#FCB638', '#6B0082'])))
# cm_purple = ScaledColormap(linear_colors(linearize_palette(np.vstack([[0.95, 0.95, 0.95], colors_purple2(np.linspace(0, 1, 15))[:,:3]]))), break_in = 0.15, break_out = 0.15)
# cm_purple_scatter2 = ScaledColormap(linear_colors(linearize_palette(np.vstack([[0.85, 0.85, 0.85], colors_purple2(np.linspace(0, 1, 15))[:,:3]]))), break_in = 0.15, break_out = 0.15)

# cm_test2 = linear_colors(list(map(to_rgb, ["#FFFD5F", "#FFD750", "#CAE200", "#700013"])))
# cm_test2 = list(map(to_rgb, ["#FFFE8F", '#F2AE36', '#72008B']))
# cm_expression = linear_colors(linearize_palette(cm_test2), start_col = [0.85, 0.85, 0.85])

# pal_prob = linear_colors(list(map(to_rgb, ['#F6EF37', '#53AD65', '#007D94'])))
# cm_prob = linear_colors(linearize_palette(pal_prob(np.linspeace(0, 1, 10)[:,:3]))

# cm_topic = matplotlib.colors.LinearSegmentedColormap("Name",
#         colors_at_timelist(map(to_rgb(['#FAE622', '#1FA188', '#1142AF'])))
#         )


# # Black -> purple (87, 16, 110) @ 0.8/3.0 -> Orange (231, 95, 44) @ 2/3 -> Yellow (248, 251, 156) @ 1
# inferno_tweaked = matplotlib.colors.LinearSegmentedColormap('InfernoTweaked',
#         colors_at_time([
#             (0, 0, 0, 0),
#             # (0.05, 87, 16, 110),
#             # (0.05, 152, 70, 179),
#             (0.1, 119, 28, 145),
#             (0.5, 231, 95, 44),
#             (1, 248, 251, 156)
#             ]))

# winter_tweaked = matplotlib.colors.LinearSegmentedColormap('InfernoTweaked',
#         colors_at_time([
#             (0, 0, 0, 255),
#             # (0.5, 0, 194, 198),
#             (0.5, 0, 120, 120),
#             (1, 0, 200, 0)
#             ]))

# black_red = matplotlib.colors.LinearSegmentedColormap('BlackRed',
#         colors_at_time([
#             (0, 0, 0, 0),
#             (1, 200, 0, 0)
#             ]))

# gray_purple = matplotlib.colors.LinearSegmentedColormap('GrayPurple',
#         colors_at_time([
#             (0, 0.9*256, 0.9*256, 0.9*256),
#             (0.25, 255, 103, 216),
#             # (0.25, 0, 0, 255),
#             (1, 0, 0, 170)
#             ]))

# gray_purple = linearize_cmap(gray_purple)

# gray_val = 0.9
# black_gray = matplotlib.colors.LinearSegmentedColormap('BlackGray', {
#         'red' : (
#             (0, 0, 0),
#             (0.5, 0, gray_val),
#             (1, gray_val, gray_val)
#             ),
#         'green' : (
#             (0, 0, 0),
#             (0.5, 0, gray_val),
#             (1, gray_val, gray_val)
#             ),
#         'blue' : (
#             (0, 0, 0),
#             (0.5, 0, gray_val),
#             (1, gray_val, gray_val)
#             )
#     })
