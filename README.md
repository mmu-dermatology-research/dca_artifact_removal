# artifact_removal

## Generating the dca split dataset
1. Open "./Modules/create_balanced_dca_dataset.py" module
2. Read through __docstring__ for module carefully - changing filepaths as necessary
3. Execute the module

**Full Model Performances on all individual testing sets:**
<table>
	<tr>
		<td>Model Used</td>
    <td>Test Set</td>
    <td colspan="3">Metrics</td>
    <td colspan="3">Micro-Average</td>
	</tr>
	<tr>
<td> </td><td> </td><td>Acc</td><td>TPR</td><td>TNR</td><td>F1</td><td>AUC</td><td>Precision</td>
</tr>
<tr><td>Clean</td><td>base-small</td><td>0.59</td><td>0.86</td><td>0.32</td><td>0.68</td><td>0.63</td><td>0.56</td></tr>
<tr><td></td><td>ns-small</td><td>0.59</td><td>0.86</td><td>0.31</td><td>0.68</td><td>0.62</td><td>0.56</td></tr>
<tr><td></td><td>telea-small</td><td>0.59</td><td>0.86</td><td>0.31</td><td>0.68</td><td>0.62</td><td>0.56</td></tr>
<tr><td> </td><td>base-medium</td><td>0.57</td><td>0.91</td><td>0.24</td><td>0.68</td><td>0.64</td><td>0.54</td></tr>
<tr><td> </td><td>ns-medium</td><td>0.62</td><td>0.88</td><td>0.36</td><td>0.70</td><td>0.68</td><td>0.58</td></tr>
<tr><td> </td><td>telea-medium</td><td>0.62</td><td>0.87</td><td>0.36</td><td>0.69</td><td>0.68</td><td>0.58</td></tr>
<tr><td> </td><td>base-large</td><td>0.51</td><td>0.99</td><td>0.01</td><td>0.67</td><td>0.58</td><td>0.50</td></tr>
<tr><td> </td><td>ns-large</td><td>0.64</td><td>0.85</td><td>0.44</td><td>0.70</td><td>0.71</td><td>0.60</td></tr>
<tr><td> </td><td>telea-large</td><td>0.65</td><td>0.85</td><td>0.45</td><td>0.71</td><td>0.71</td><td>0.61</td></tr>
<tr><td> </td><td>base-oth</td><td>0.58</td><td>0.90</td><td>0.26</td><td>0.67</td><td>0.65</td><td>0.55</td></tr>
<tr><td> </td><td>ns-oth</td><td>0.58</td><td>0.87</td><td>0.29</td><td>0.67</td><td>0.66</td><td>0.55</td></tr>
<tr><td> </td><td>telea-oth</td><td>0.58</td><td>0.87</td><td>0.29</td><td>0.67</td><td>0.66</td><td>0.55</td></tr>
<tr><td>Binary DCA</td><td>base-small</td><td>0.61</td><td>0.90</td><td>0.33</td><td>0.70</td><td>0.67</td><td>0.57</td></tr>
<tr><td></td><td>ns-small</td><td>0.61</td><td>0.89</td><td>0.33</td><td>0.70</td><td>0.67</td><td>0.57</td></tr>
<tr><td></td><td>telea-small</td><td>0.61</td><td>0.89</td><td>0.33</td><td>0.70</td><td>0.67</td><td>0.57</td></tr>
<tr><td> </td><td>base-medium</td><td>0.63</td><td>0.94</td><td>0.31</td><td>0.72</td><td>0.68</td><td>0.58</td></tr>
<tr><td> </td><td>ns-medium</td><td>0.65</td><td>0.85</td><td>0.44</td><td>0.71</td><td>0.73</td><td>0.60</td></tr>
<tr><td> </td><td>telea-medium</td><td>0.65</td><td>0.85</td><td>0.45</td><td>0.70</td><td>0.73</td><td>0.61</td></tr>
<tr><td> </td><td>base-large</td><td>0.55</td><td>0.96</td><td>0.13</td><td>0.68</td><td>0.62</td><td>0.53</td></tr>
<tr><td> </td><td>ns-large</td><td>0.70</td><td>0.79</td><td>0.61</td><td>0.73</td><td>0.75</td><td>0.67</td></tr>
<tr><td> </td><td>telea-large</td><td>0.70</td><td>0.78</td><td>0.61</td><td>0.72</td><td>0.75</td><td>0.67</td></tr>
<tr><td> </td><td>base-oth</td><td>0.60</td><td>0.83</td><td>0.36</td><td>0.67</td><td>0.67</td><td>0.57</td></tr>
<tr><td> </td><td>ns-oth</td><td>0.60</td><td>0.82</td><td>0.39</td><td>0.67</td><td>0.68</td><td>0.57</td></tr>
<tr><td> </td><td>telea-oth</td><td>0.60</td><td>0.82</td><td>0.39</td><td>0.67</td><td>0.68</td><td>0.57</td></tr>
<tr><td>Realistic DCA</td><td>base-small</td><td>0.60</td><td>0.85</td><td>0.35</td><td>0.68</td><td>0.65</td><td>0.57</td></tr>
<tr><td></td><td>ns-small</td><td>0.60</td><td>0.85</td><td>0.35</td><td>0.68</td><td>0.66</td><td>0.57</td></tr>
<tr><td></td><td>telea-small</td><td>0.60</td><td>0.84</td><td>0.36</td><td>0.68</td><td>0.66</td><td>0.57</td></tr>
<tr><td> </td><td>base-medium</td><td>0.64</td><td>0.75</td><td>0.53</td><td>0.68</td><td>0.70</td><td>0.62</td></tr>
<tr><td> </td><td>ns-medium</td><td>0.66</td><td>0.84</td><td>0.48</td><td>0.71</td><td>0.72</td><td>0.62</td></tr>
<tr><td> </td><td>telea-medium</td><td>0.66</td><td>0.82</td><td>0.49</td><td>0.71</td><td>0.73</td><td>0.62</td></tr>
<tr><td> </td><td>base-large</td><td>0.60</td><td>0.39</td><td>0.80</td><td>0.49</td><td>0.63</td><td>0.66</td></tr>
<tr><td> </td><td>ns-large</td><td>0.66</td><td>0.70</td><td>0.63</td><td>0.68</td><td>0.74</td><td>0.65</td></tr>
<tr><td> </td><td>telea-large</td><td>0.67</td><td>0.69</td><td>0.65</td><td>0.67</td><td>0.74</td><td>0.66</td></tr>
<tr><td> </td><td>base-oth</td><td>0.58</td><td>0.81</td><td>0.35</td><td>0.66</td><td>0.65</td><td>0.55</td></tr>
<tr><td> </td><td>ns-oth</td><td>0.58</td><td>0.79</td><td>0.37</td><td>0.65</td><td>0.65</td><td>0.56</td></tr>
<tr><td> </td><td>telea-oth</td><td>0.58</td><td>0.79</td><td>0.37</td><td>0.65</td><td>0.65</td><td>0.56</td></tr>
</table>
