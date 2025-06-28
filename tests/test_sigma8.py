import cosmology_amilly.sigma8 as s8

red0 = s8.Power_Spectrum(0, "tests/test_data_source/camb_0.dat")
red1 = s8.Power_Spectrum(1, "tests/test_data_source/camb_1.dat")
red2 = s8.Power_Spectrum(2, "tests/test_data_source/camb_2.dat")
red3 = s8.Power_Spectrum(3, "tests/test_data_source/camb_3.dat")

print(red0.sigma_8())
print(red1.sigma_8())
print(red2.sigma_8())
print(red3.sigma_8())
