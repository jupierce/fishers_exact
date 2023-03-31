import multiprocessing
import traceback
from concurrent.futures import ThreadPoolExecutor

from multiprocessing import Process

from typing import Dict, NamedTuple, List, Tuple, Any, Optional, Iterable, Set, Union

from django.utils.html import format_html
from django.http import HttpResponse
import django_tables2 as tables

from .bq_junit import Junit, select, sum, count, any_value, EnvironmentModel, EnvironmentTestRecords, EnvironmentName, TestRecordAssessment, ComponentTestRecords, TestName, TestRecord, TestId, AggregateTestAssessment
from .fishers import fisher_significant

import fast_fisher.fast_fisher_cython

from sqlalchemy import text

from django.shortcuts import render

ProwjobName = str


def index(request):
    return render(request, "main/index.html")


class ImageColumnLink(NamedTuple):
    image_path: str
    title: Optional[str] = None
    height: Optional[int] = None
    width: Optional[int] = None
    href: Optional[str] = None
    href_params: Optional[Dict[str, str]] = None


class AssessmentImageColumnLink:
    def __init__(self, assessment: Union[TestRecordAssessment, AggregateTestAssessment], href: str = '/main/report', href_params: Dict[str, str] = None):
        self.image_path = f'main/{assessment.image_path}'
        self.height = 16
        self.width = 16
        if assessment.val < 0 and isinstance(assessment, AggregateTestAssessment):
            if assessment.count == 1:
                self.width = 8
                self.height = 8
            elif assessment.count < 4:
                self.width = 11
                self.height = 11
        self.href = href
        self.href_params = href_params
        self.title = assessment.description


class ImageColumn(tables.Column):

    def render(self, value: ImageColumnLink):
        image_path = value.image_path

        height_attr = ''
        if value.height:
            height_attr = f'height="{value.height}" '

        width_attr = ''
        if value.width:
            width_attr = f'width="{value.height}" '

        title_attr = ''
        if value.title:
            title_attr = f'title="{value.title}" '

        content = f'<img {title_attr}{height_attr}{width_attr} src="/static/{image_path}"></img>'
        if value.href:
            content = f'<a href="{value.href}?{dict_to_params_url(value.href_params)}">{content}</a>'
        return format_html(content)


def dict_to_params_url(params):
    if not params:
        return ''
    return '&'.join([f'{key}={value}' for key, value in params.items()])


def _render_prowjob_rows(rows) -> str:
    result = ''
    char_count = 0
    for row in rows:
        # Some files have multiple successes and failures that are not counted as flakes.
        # An example junit I found had a success of a specific test and then a failure
        # of the same test which came later.
        # ref: [sig-arch] Check if alerts are firing during or after upgrade success
        # So each row may need to be represented by multiple characters.
        # Use max() to provide sensible results if the time selection in the query
        # has only selected part of a file.
        failure_iterations = max(0, row['total_count'] - row['flake_count'] - row['success_count'])
        flake_iterations = row['flake_count']
        success_iterations = max(0, row['success_count'] - row['flake_count'])
        artifacts_split = row["file_path"].split('/artifacts/', 1)
        spyglass_path = 'https://prow.ci.openshift.org/view/gs/origin-ci-test/' + artifacts_split[0]
        junit_file_path = '' if len(artifacts_split) == 1 else artifacts_split[1]

        char_entries: List[str] = (['S'] * success_iterations) + (['s'] * flake_iterations) + (['F'] * failure_iterations)
        for outcome_char in char_entries:
            result += f'<a class="outcome_{outcome_char}" href="{spyglass_path}/" title="{junit_file_path}" alt="{junit_file_path}">{outcome_char}</a> '
            char_count += 1
            if char_count % 20 == 0:
                result += '<br>'
    return format_html(result)


class ProwjobTable(tables.Table):
    prowjob_name = tables.Column()
    basis_info = tables.Column()
    basis_runs = tables.Column()
    sample_info = tables.Column()
    sample_runs = tables.Column()
    statistically_significant = tables.Column()

    class Meta:
        attrs = {"class": "paleblue"}


class AllComponentsTable(tables.Table):
    name = tables.Column()

    def __init__(self, data, extra_columns, new_key=None, params=None):
        super().__init__(data, extra_columns=extra_columns)
        self.params = params
        self.new_key = new_key

    class Meta:
        attrs = {"class": "paleblue"}

    def render_name(self, value):
        if self.params is None:
            params = dict()
        else:
            params = dict(self.params)
        if self.new_key:
            params[self.new_key] = value

        if params:
            params_str = dict_to_params_url(params)
            return format_html(f'<a href="/main/report?{params_str}">{value}</a>')
        else:
            return value


def report(request):

    def insufficient_sanitization(parameter_name: str, parameter_default: Optional[str] = None) -> Optional[str]:
        val = request.GET.get(parameter_name, parameter_default)
        if val is not None and ("'" in val or '"' in val or '\\' in val):
            raise IOError('Sanitization failure')
        return val

    basis_release = insufficient_sanitization('basis_release')
    basis_start_dt = insufficient_sanitization('basis_start_dt')
    basis_end_dt = insufficient_sanitization('basis_end_dt')

    sample_release = insufficient_sanitization('sample_release')
    sample_start_dt = insufficient_sanitization('sample_start_dt')
    sample_end_dt = insufficient_sanitization('sample_end_dt')

    confidence_param = insufficient_sanitization("confidence", "95")
    fisher_alpha: float = (100 - int(confidence_param)) / 100

    pity_param = insufficient_sanitization("pity", "5")
    pity_factor: float = int(pity_param) / 100

    missing_samples_param = insufficient_sanitization("missing", "ok")
    regression_when_missing = missing_samples_param != "ok"

    include_disruptions_param = insufficient_sanitization("disruption", "0")
    include_disruptions = include_disruptions_param == 1

    target_component_name = insufficient_sanitization('component', None)
    target_capability_name = insufficient_sanitization('capability', None)
    target_test_id = insufficient_sanitization('test_id', None)
    target_test_uuid = insufficient_sanitization('test_uuid', None)
    target_platform_name = insufficient_sanitization('platform', None)
    target_upgrade_name = insufficient_sanitization('upgrade', None)
    target_arch_name = insufficient_sanitization('arch', None)
    target_variant_name = insufficient_sanitization('variant', None)
    target_network_name = insufficient_sanitization('network', None)
    target_environment_name = insufficient_sanitization('environment', None)

    group_by_param = insufficient_sanitization('group_by', None)

    exclude_platforms_param = insufficient_sanitization('exclude_platforms', None)
    exclude_arches_param = insufficient_sanitization('exclude_arches', None)
    exclude_networks_param = insufficient_sanitization('exclude_networks', None)
    exclude_upgrades_param = insufficient_sanitization('exclude_upgrades', None)
    exclude_variants_param = insufficient_sanitization('exclude_variants', None)

    j = Junit
    basis_environment_model = EnvironmentModel('basis', group_by_param)
    sample_environment_model = EnvironmentModel('sample', group_by_param)
    pqb = basis_environment_model.get_environment_query_scan()

    def assert_all_set(lt: Iterable, error_msg: str):
        if not all(lt):
            raise ValueError(error_msg)

    assert_all_set((basis_start_dt, basis_end_dt, basis_release), 'At least one basis coordinate has not been specified')

    assert_all_set((sample_start_dt, sample_end_dt, sample_release), 'At least one sample coordinate has not been specified')

    context = {
        'basis_release': basis_release,
        'basis_start_dt': basis_start_dt,
        'basis_end_dt': basis_end_dt,
        'sample_release': sample_release,
        'sample_start_dt': sample_start_dt,
        'sample_end_dt': sample_end_dt,
        'group_by': group_by_param,
    }

    if target_upgrade_name:
        pqb = pqb.filter(
            j.upgrade == target_upgrade_name
        )

    if target_arch_name:
        pqb = pqb.filter(
            j.arch == target_arch_name
        )

    if target_network_name:
        pqb = pqb.filter(
            j.network == target_network_name
        )

    if target_platform_name:
        pqb = pqb.filter(
            j.platform == target_platform_name
        )

    if not include_disruptions:
        pqb = pqb.filter(
            j.test_name.notlike('%disruption/%')
        )

    if target_test_id:
        pqb = pqb.filter(
            j.test_id == target_test_id
        )

    if exclude_platforms_param:
        context['exclude_platforms'] = exclude_platforms_param
        for exclude_element in exclude_platforms_param.split(','):
            if not exclude_element:
                continue
            pqb = pqb.filter(
                j.platform.notlike(f'{exclude_element}%')
            )

    if exclude_arches_param:
        context['exclude_arches'] = exclude_arches_param
        for exclude_name in exclude_arches_param.split(','):
            if not exclude_name:
                continue
            pqb = pqb.filter(
                j.arch != exclude_name
            )

    if exclude_networks_param:
        context['exclude_networks'] = exclude_networks_param
        for exclude_name in exclude_networks_param.split(','):
            if not exclude_name:
                continue
            pqb = pqb.filter(
                j.network != exclude_name
            )

    if exclude_upgrades_param:
        context['exclude_upgrades'] = exclude_upgrades_param
        upgrade_name_db_mapping = {
            'install': 'none',
            'minor': 'upgrade-minor',
            'micro': 'upgrade-micro',
        }
        for exclude_name in exclude_upgrades_param.split(','):
            if not exclude_name or exclude_name not in upgrade_name_db_mapping:
                continue
            pqb = pqb.filter(
                j.upgrade != upgrade_name_db_mapping[exclude_name]
            )

    if exclude_variants_param:
        context['exclude_variants'] = exclude_variants_param
        for exclude_element in exclude_variants_param.split(','):
            if not exclude_element:
                continue
            pqb = pqb.filter(
                j.flat_variants.notlike(f'%{exclude_element}%')
            )

    # q = f'''
    #     SELECT CONCAT(network, " ", upgrade, " ", arch, " ", platform) as nurp, test_id, ANY_VALUE(test_name) as test_name, COUNT(*) as total_count, SUM(success_val) as success_count, SUM(flake_count) as flake_count
    #     FROM `openshift-gce-devel.ci_analysis_us.junit`
    #     WHERE modified_time >= DATETIME(TIMESTAMP "{basis_start_dt}:00+00") AND modified_time < DATETIME(TIMESTAMP "{basis_end_dt}:00+00") {target_nurp_filter} {target_test_filter}
    #           AND branch = "{basis_release}" AND file_path NOT LIKE "%/junit_operator.xml" GROUP BY network, upgrade, arch, platform, test_id
    # '''
    basis_query = pqb.filter(
        j.modified_time >= j.format_modified_time(basis_start_dt),
        j.modified_time < j.format_modified_time(basis_end_dt),
        j.branch == basis_release
    )

    # q = f'''
    #     SELECT CONCAT(network, " ", upgrade, " ", arch, " ", platform) as nurp, test_id, ANY_VALUE(test_name) as test_name, COUNT(*) as total_count, SUM(success_val) as success_count, SUM(flake_count) as flake_count
    #     FROM `openshift-gce-devel.ci_analysis_us.junit`
    #     WHERE modified_time >= DATETIME(TIMESTAMP "{sample_start_dt}:00+00") AND modified_time < DATETIME(TIMESTAMP "{sample_end_dt}:00+00") {target_nurp_filter} {target_test_filter}
    #           AND branch = "{sample_release}" AND file_path NOT LIKE "%/junit_operator.xml" GROUP BY network, upgrade, arch, platform, test_id
    # '''

    sample_query = pqb.filter(
        j.modified_time >= j.format_modified_time(sample_start_dt),
        j.modified_time < j.format_modified_time(sample_end_dt),
        j.branch == sample_release
    )

    # Performance note: the .query() calls return relatively quickly. Bigquery is doing the calculations
    # and python will only block when you try to read rows that haven't been calculated yet.
    # Thus, the majority of the time spent will usually be in categorize where the data is actually being
    # read.
    # Based on simple profiling, it looks like data normally starts to stream in fairly quickly, but
    # getting to the end takes awhile. So categorize gets its first records in a few seconds, but
    # runs for 10 times that long despite not doing much computation itself (i.e. it is blocking
    # waiting for bigquery).

    executor = ThreadPoolExecutor(2)
    basis_future = executor.submit(basis_environment_model.read_in_query, basis_query)
    sample_future = executor.submit(sample_environment_model.read_in_query, sample_query)
    basis_future.result()
    sample_future.result()

    sample_environment_model.build_mass_assessment_cache(basis_environment_model, alpha=fisher_alpha, regression_when_missing=regression_when_missing, pity_factor=pity_factor)

    ordered_environment_names: List[EnvironmentName] = sorted(set(list(sample_environment_model.get_ordered_environment_names()) + list(basis_environment_model.get_ordered_environment_names())))

    if target_platform_name:
        context['platform'] = target_platform_name
    if target_network_name:
        context['network'] = target_network_name
    if target_upgrade_name:
        context['upgrade'] = target_upgrade_name
    if target_arch_name:
        context['arch'] = target_arch_name
    if target_variant_name:
        context['variant'] = target_variant_name

    if confidence_param != "95":
        context['confidence'] = confidence_param

    if pity_param != "5":
        context['pity'] = pity_param

    if missing_samples_param != "ok":
        context['missing'] = missing_samples_param

    if include_disruptions:
        context['disruption'] = include_disruptions_param

    def populate_environment_link_context(environment_test_records: EnvironmentTestRecords, link_context: Dict):
        if environment_test_records.platform:
            link_context['platform'] = environment_test_records.platform
        if environment_test_records.network:
            link_context['network'] = environment_test_records.network
        if environment_test_records.upgrade:
            link_context['upgrade'] = environment_test_records.upgrade
        if environment_test_records.arch:
            link_context['arch'] = environment_test_records.arch
        if environment_test_records.variant:
            link_context['variant'] = environment_test_records.variant

    if target_test_id and not target_test_uuid:  # Rendering a specifically requested TestRecordSet test_id
        if not target_environment_name:
            return HttpResponse(f'No environment parameter was specified')
        if not target_component_name:
            return HttpResponse(f'No component parameter was specified')
        if not target_capability_name:
            return HttpResponse(f'No capability parameter was specified')

        sample_test_record_set = sample_environment_model.get_environment_test_records(target_environment_name).get_component_test_records(target_component_name).get_capability_test_records(target_capability_name).get_test_record_set(target_test_id)
        context['environment'] = target_environment_name
        context['component'] = target_component_name
        context['capability'] = target_capability_name
        context['test_id'] = target_test_id

        uuid_count = len(sample_test_record_set.test_records)
        if uuid_count > 1:
            # There are more than one test uuids that have been grouped into this test record set,
            # so provide a UI that allows the user to view each UUID and drill to the one they
            # want more information for.

            test_record_summary: List[Dict] = list()
            extra_columns = [
                ('platform', tables.Column()),
                ('arch', tables.Column()),
                ('network', tables.Column()),
                ('upgrade', tables.Column()),
                ('variant', tables.Column()),
                ('status', ImageColumn())
            ]

            for test_record in sorted(sample_test_record_set.get_test_records(), key=lambda x: x.test_uuid):

                env_attributes: Dict = {
                    'platform': test_record.platform,
                    'arch': test_record.arch,
                    'network': test_record.network,
                    'upgrade': test_record.upgrade,
                    'variant': test_record.flat_variants,
                }
                href_params = dict(context)
                href_params.update(env_attributes)
                href_params['test_uuid'] = test_record.test_uuid

                row = {
                    'name': test_record.test_name,
                    'status': AssessmentImageColumnLink(
                        test_record.cached_assessment,
                        href_params=dict(href_params)
                    )
                }
                row.update(env_attributes)

                test_record_summary.append(row)

            table = AllComponentsTable(test_record_summary,
                                       extra_columns=extra_columns,
                                       )
            context['table'] = table
            context['breadcrumb'] = f'{target_environment_name} > {target_component_name} > {target_capability_name} > Disambiguate'
            return render(request, 'main/report-table.html', context)

        elif uuid_count == 1:
            # There is only one test uuid associated with this test record set. Send the user
            # straight to the UI for the specific uuid.
            target_test_uuid = sample_test_record_set.get_test_record_uuids()[0]
        else:
            return HttpResponse('No tests are associated with this environment and test id')

    if target_test_uuid:
        if not target_environment_name:
            return HttpResponse(f'No environment parameter was specified')
        if not target_component_name:
            return HttpResponse(f'No component parameter was specified')
        if not target_capability_name:
            return HttpResponse(f'No capability parameter was specified')
        if not target_test_id:
            return HttpResponse(f'No test_id parameter was specified')

        sample_test_record_set = sample_environment_model.get_environment_test_records(
            target_environment_name).get_component_test_records(target_component_name).get_capability_test_records(
            target_capability_name).get_test_record_set(target_test_id)
        sample_test_record = sample_test_record_set.get_test_record(target_test_uuid)

        if not sample_test_record:
            return HttpResponse('No sample test record found for test uuid')

        basis_test_record_set = basis_environment_model.get_environment_test_records(
            target_environment_name).get_component_test_records(target_component_name).get_capability_test_records(
            target_capability_name).get_test_record_set(target_test_id)

        basis_test_record = basis_test_record_set.get_test_record(target_test_uuid)

        if not basis_test_record:
            basis_test_record = TestRecord(
                env_name=sample_test_record.env_name,
                platform=sample_test_record.platform,
                network=sample_test_record.network,
                upgrade=sample_test_record.upgrade,
                arch=sample_test_record.arch,
                flat_variants=sample_test_record.flat_variants,
                test_id=sample_test_record.test_id,
                test_name=sample_test_record.test_name,
                testsuite=sample_test_record.testsuite,
            )

        context['sample_test'] = sample_test_record
        context['basis_test'] = basis_test_record
        context['fishers_exact'] = str(fast_fisher.fast_fisher_cython.fisher_exact(
            sample_test_record.failure_count, sample_test_record.success_count,
            basis_test_record.failure_count, basis_test_record.success_count,
            alternative='greater' if sample_test_record.assessment() != TestRecordAssessment.SIGNIFICANT_IMPROVEMENT else 'less'
        ))

        base_test_query = select(
            j.file_path,
            any_value(j.prowjob_name).label('prowjob_name'),
            sum(j.success_val).label('success_count'),
            sum(j.flake_count).label('flake_count'),
            count('*').label('total_count'),  # Including flakes
        ).where(
            j.platform == sample_test_record.platform,
            j.network == sample_test_record.network,
            j.upgrade == sample_test_record.upgrade,
            j.arch == sample_test_record.arch,
            j.flat_variants == sample_test_record.flat_variants,
            j.test_id == sample_test_record.test_id
        ).group_by(
            j.file_path,
            j.modified_time
        ).order_by(
            j.modified_time  # Show test runs in roughly chronological order
        )

        basis_test_query = base_test_query.where(
            j.modified_time >= j.format_modified_time(basis_start_dt),
            j.modified_time < j.format_modified_time(basis_end_dt),
            j.branch == basis_release
        )

        sample_test_query = base_test_query.where(
            j.modified_time >= j.format_modified_time(sample_start_dt),
            j.modified_time < j.format_modified_time(sample_end_dt),
            j.branch == sample_release
        )

        # Aggregate test successes/failures by prowjob
        prowjob_names = set()

        basis_prowjob_runs: Dict[ProwjobName, List] = dict()
        for row in basis_test_query.execute():
            prowjob_name: str = row['prowjob_name']
            prowjob_name = prowjob_name.replace(basis_release, 'X.X')  # Strip release specific information from prowjob name
            prowjob_names.add(prowjob_name)
            if prowjob_name not in basis_prowjob_runs:
                basis_prowjob_runs[prowjob_name] = list()
            basis_prowjob_runs[prowjob_name].append(row)

        sample_prowjob_runs: Dict[ProwjobName, List] = dict()
        for row in sample_test_query.execute():
            prowjob_name: str = row['prowjob_name']
            prowjob_name = prowjob_name.replace(sample_release, 'X.X')  # Strip release specific information from prowjob name
            prowjob_names.add(prowjob_name)
            if prowjob_name not in sample_prowjob_runs:
                sample_prowjob_runs[prowjob_name] = list()
            sample_prowjob_runs[prowjob_name].append(row)

        prowjob_analysis: List[Dict[str, str]] = list()
        for prowjob_name in sorted(prowjob_names):

            basis_prowjob_rows = basis_prowjob_runs.get(prowjob_name, list())
            basis_success_count = 0
            basis_flake_count = 0
            basis_total_count = 0

            for basis_prowjob_row in basis_prowjob_rows:
                basis_success_count += basis_prowjob_row['success_count']
                basis_flake_count += basis_prowjob_row['flake_count']
                basis_total_count += basis_prowjob_row['total_count']  # Includes flakes
            basis_failure_count = max(0, basis_total_count-basis_flake_count-basis_success_count)
            basis_total_minus_flakes = (basis_total_count - basis_flake_count)
            basis_success_rate = '{:.2f}'.format(0.0 if basis_total_minus_flakes == 0 else 100 * basis_success_count / basis_total_minus_flakes)

            sample_prowjob_rows = sample_prowjob_runs.get(prowjob_name, list())
            sample_success_count = 0
            sample_flake_count = 0
            sample_total_count = 0

            for sample_prowjob_row in sample_prowjob_rows:
                sample_success_count += sample_prowjob_row['success_count']
                sample_flake_count += sample_prowjob_row['flake_count']
                sample_total_count += sample_prowjob_row['total_count']  # Includes flakes
            sample_failure_count = max(0, sample_total_count-sample_flake_count-sample_success_count)
            sample_total_minus_flakes = (sample_total_count - sample_flake_count)
            sample_success_rate = '{:.2f}'.format(0.0 if sample_total_minus_flakes == 0 else 100 * sample_success_count / sample_total_minus_flakes)

            prowjob_analysis.append(
                {
                    'prowjob_name': prowjob_name,
                    'basis_info': format_html(f'rate={basis_success_rate}%<br>successes={basis_success_count}<br>failures={basis_failure_count}<br>flakes={basis_flake_count}'),
                    'basis_runs': _render_prowjob_rows(basis_prowjob_rows),
                    'sample_info': format_html(f'rate={sample_success_rate}%<br>successes={sample_success_count}<br>failures={sample_failure_count}<br>flakes={sample_flake_count}'),
                    'sample_runs': _render_prowjob_rows(sample_prowjob_rows),
                    'statistically_significant': fisher_significant(
                        sample_failure_count, sample_success_count,
                        basis_failure_count, basis_success_count,
                        alpha=fisher_alpha,
                    ),
                }
            )

        prowjob_table = ProwjobTable(data=prowjob_analysis)
        context['table'] = prowjob_table
        context['breadcrumb'] = f'{target_environment_name} > {target_component_name} > {target_capability_name} > {sample_test_record.test_name}'
        return render(request, 'main/report-test.html', context)

    if not target_component_name:  # Rendering all components
        component_summary: List[Dict] = list()
        extra_columns = []
        all_component_names = set()

        for environment_name in ordered_environment_names:
            image_href_params = dict(context)
            image_href_params['environment'] = environment_name
            extra_columns.append((environment_name, ImageColumn()))
            sample_environment_test_records = sample_environment_model.get_environment_test_records(environment_name)
            all_component_names.update(sample_environment_test_records.get_component_names())

        for component_name in sorted(list(all_component_names)):
            if 'sig' not in component_name:
                continue

            row = {
                'name': component_name,
            }

            for environment_name in ordered_environment_names:
                sample_environment_test_records: EnvironmentTestRecords = sample_environment_model.get_environment_test_records(environment_name)
                sample_component_test_records = sample_environment_test_records.get_component_test_records(component_name)

                component_assessment: AggregateTestAssessment = sample_component_test_records.assessment()

                href_params = dict(context)
                href_params['component'] = component_name
                href_params['environment'] = environment_name
                populate_environment_link_context(sample_environment_test_records, href_params)

                row[environment_name] = AssessmentImageColumnLink(
                    component_assessment,
                    href_params=dict(href_params)
                )

            component_summary.append(row)

        table = AllComponentsTable(component_summary,
                                   extra_columns=extra_columns,
                                   new_key='component',
                                   params=dict(context),
                                   )
        context['table'] = table
        context['breadcrumb'] = f'All Components'
        return render(request, 'main/report-table.html', context)
    else:  # Rendering all of a specific component's capabilities

        context['component'] = target_component_name

        if not target_environment_name:  # Rendering a component or capability with environments as column heading

            if target_capability_name:
                extra_columns = []

                test_id_lookup: Dict[TestName, TestId] = dict()
                for environment_name in sample_environment_model.get_ordered_environment_names():
                    image_href_params = dict(context)
                    image_href_params['environment'] = environment_name
                    extra_columns.append((environment_name, ImageColumn()))

                    # It's unlikely but possible that a component in one environment has different capabilities
                    # than a component in another environment. Build a full set of names across environments.
                    for sample_test_record_set in sample_environment_model.get_environment_test_records(environment_name).get_component_test_records(target_component_name).get_capability_test_records(target_capability_name).get_test_record_sets():
                        test_id_lookup[sample_test_record_set.canonical_test_name] = sample_test_record_set.test_id

                test_summary: List[Dict] = list()

                for test_name in sorted(list(test_id_lookup.keys())):
                    test_record_set_test_id = test_id_lookup[test_name]

                    row = {
                        'name': test_name,
                    }

                    for environment_name in sample_environment_model.get_ordered_environment_names():
                        sample_environment_test_records = sample_environment_model.get_environment_test_records(environment_name)
                        assessment: TestRecordAssessment = sample_environment_test_records.get_component_test_records(target_component_name).get_capability_test_records(target_capability_name).get_test_record_set(test_record_set_test_id).assessment()
                        href_params = dict(context)
                        href_params['capability'] = target_capability_name
                        href_params['test_id'] = test_record_set_test_id
                        href_params['environment'] = environment_name
                        populate_environment_link_context(sample_environment_test_records, href_params)
                        row[environment_name] = AssessmentImageColumnLink(
                            assessment,
                            href_params=href_params,
                        )

                    test_summary.append(row)

                table = AllComponentsTable(test_summary,
                                           extra_columns=extra_columns,
                                           )
                context['table'] = table
                context['breadcrumb'] = f'{target_component_name} > {target_capability_name}'
                return render(request, 'main/report-table.html', context)

            else:  # Rending the capabilities of a component vs environment columns
                extra_columns = []

                # It's remote, but possible that the same component within different
                # environments have different capabilities. Develop a
                # set of all names across each environment.
                capability_names: Set[str] = set()

                for environment_name in sample_environment_model.get_ordered_environment_names():
                    image_href_params = dict(context)
                    image_href_params['environment'] = environment_name
                    extra_columns.append((environment_name, ImageColumn()))
                    capability_names.update(
                        sample_environment_model.get_environment_test_records(environment_name).get_component_test_records(target_component_name).get_capability_names()
                    )

                capability_summary: List[Dict] = list()
                for capability_name in sorted(capability_names):

                    row = {
                        'name': capability_name,
                    }

                    for environment_name in ordered_environment_names:
                        sample_environment_test_records = sample_environment_model.get_environment_test_records(environment_name)
                        assessment = sample_environment_test_records.get_component_test_records(target_component_name).get_capability_test_records(capability_name).assessment()
                        href_params = dict(context)
                        href_params['capability'] = capability_name
                        href_params['environment'] = environment_name
                        populate_environment_link_context(sample_environment_test_records, href_params)
                        row[environment_name] = AssessmentImageColumnLink(
                            assessment,
                            href_params=href_params,
                        )

                    capability_summary.append(row)

                table = AllComponentsTable(capability_summary,
                                           extra_columns=extra_columns,
                                           new_key='capability',
                                           params=dict(context)
                                           )
                context['table'] = table
                context['breadcrumb'] = f'{target_component_name}'
                return render(request, 'main/report-table.html', context)

        else:  # Showing capabilities for specific component in specific environment
            context['environment'] = target_environment_name
            sample_environment_test_records = sample_environment_model.get_environment_test_records(target_environment_name)
            sample_component_test_records = sample_environment_test_records.get_component_test_records(target_component_name)

            if not target_capability_name:  # Show capabilities of a specific component in a specific environment
                capability_summary: List[Dict] = list()
                for capability_name in sorted(sample_component_test_records.get_capability_names()):
                    sample_capability_test_records = sample_component_test_records.get_capability_test_records(capability_name)
                    assessment = sample_capability_test_records.assessment()
                    href_params = dict(context)
                    href_params['capability'] = capability_name
                    populate_environment_link_context(sample_environment_test_records, href_params)
                    capability_summary.append({
                        'name': capability_name,
                        'status': AssessmentImageColumnLink(
                            assessment,
                            href_params=href_params,
                        )
                    })

                table = AllComponentsTable(capability_summary, extra_columns=[('status', ImageColumn())])
                context['table'] = table
                context['breadcrumb'] = f'{target_environment_name} > {target_component_name}'
                return render(request, 'main/report-table.html', context)
            else:  # Show tests of a specific capability in a specific environment
                test_summary: List[Dict] = list()
                context['capability'] = target_capability_name
                capability_test_records = sample_component_test_records.get_capability_test_records(target_capability_name)

                for test_record_set in sorted(list(capability_test_records.get_test_record_sets()), key=lambda x: x.canonical_test_name):
                    assessment = test_record_set.assessment()
                    href_params = dict(context)
                    href_params['test_id'] = test_record_set.test_id
                    populate_environment_link_context(sample_environment_test_records, href_params)
                    test_summary.append({
                        'name': test_record_set.canonical_test_name,
                        'status': AssessmentImageColumnLink(
                            assessment,
                            href_params=href_params,
                        )
                    })

                table = AllComponentsTable(test_summary, extra_columns=[('status', ImageColumn())])
                context['table'] = table
                context['breadcrumb'] = f'{target_environment_name} > {target_component_name} > {target_capability_name}'
                return render(request, 'main/report-table.html', context)
