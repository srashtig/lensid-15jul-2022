from setuptools import setup, find_packages

setup(
    name="lensid",
    version="0.1.0",
    install_requires=open('requirements.txt').read(),
    author='Srashti Goyal, Shasvath Kapadia, P. Ajith',
    author_email='srashti.goyal@icts.res.in',
    license='International Center for Theoretical Sciences, Bangalore, India',
    packages=['lensid'],
    include_package_data=True,
       entry_points = {
        'console_scripts': [
            'lensid_fits_to_cart=lensid.utils.lensid_fits_to_cart:main',
            'lensid_create_lensed_df=lensid.injections.lensid_create_lensed_df:main',
            'lensid_create_unlensed_df=lensid.injections.lensid_create_unlensed_df:main',
            'lensid_create_lensed_inj_xmls=lensid.injections.lensid_create_lensed_inj_xmls:main'
            'lensid_create_unlensed_inj_xmls=lensid.injections.lensid_create_lensed_inj_xmls:main',
            'lensid_create_qts_lensed_injs=lensid.injections.lensid_create_qts_lensed_injs:main',
            'lensid_create_qts_unlensed_injs=lensid.injections.lensid_create_qts_unlensed_injs:main',
            'lensid_sky_injs_cart=lensid.injections.lensid_sky_injs_cart:main',
            'lensid_get_features_qts_ml=lensid.feature_extraction.lensid_get_features_qts_ml:main',
                        'lensid_get_features_sky_ml=lensid.feature_extraction.lensid_get_features_sky_ml:main'

        ]
    },
    scripts=["scripts/lensid_create_bayestar_sky_lensed_injs.sh","scripts/lensid_create_bayestar_sky_unlensed_injs.sh"]
)
