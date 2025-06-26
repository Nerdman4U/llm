PACKAGE=llm
injected_packages=(locshapython)
injected_versions=(0.7)

INSTALL_TYPE=cli_tool
ACTION=$1
VERSION=$2 # version to install or update
UPDATE_WITH_VERSION=$3 # version to update to, different from $VERSION

__gim_add_importable_venv() {
    # Add virtual environment to ~/.local/envs/<package>-<version>
    # NOTE: use only when package is to be imported from other packages.
    # Use pipx when package is run as a script.
    #
    # echo "Adding virtual environment to ~/.local/envs"
    # cd ~/.local/envs
    # python -m venv $PACKAGE-$VERSION
    echo ""
}

__gim_remove_local_egg() {
    # NOTE: with src/egg install is skipped with:
    # <package> is already installed with the same version as the provided wheel.
    # Use --force-reinstall to force an installation of the wheel.
    echo "Removing local egg-info directory"
    rm -rf "src/{${PACKAGE}.egg-info"
}

__gim_deactivate_venv() {
    # Deactivate the virtual environment
    echo "Deactivating virtual environment"
    type -t deactivate &>/dev/null && deactivate
}

__gim_install_as_importable_package() {
    # Install the package as an importable package.
    if [ -z "${PACKAGE}" ] || [ -z "${VERSION}" ]; then
        echo "Usage: __gim_install_as_importable_package <package> <version>"
        return 1
    fi
    __gim_remove_local_egg
    __gim_deactivate_venv

    echo "Activating ${PACKAGE}-${VERSION}"
    source ~/.local/envs/${PACKAGE}-${VERSION}/bin/activate

    pip install "dist/${PACKAGE}-${VERSION}-py3-none-any.whl" --force-reinstall
    echo "Installed ${PACKAGE}-${VERSION}"
}

__gim_update_cli_tool() {
    if [ -z "${PACKAGE}" ] || [ -z "${VERSION}" ] || [ -z "${UPDATE_WITH_VERSION}" ]; then
        echo "Usage: __gim_update_cli_tool <package> <version> <update_with_version>"
        return 1
    fi

    # NOTE: pipx v1.0 does not have uninject command. It does not update already injected package. So,
    # package is removed here and then injected again.
    echo "Uninjecting ${PACKAGE}${VERSION} from pipx"
    DASHED_VERSION=${VERSION//./-}
    DASHED_PACKAGE=${PACKAGE//_/-}
    RM_CMD="rm -rf ~/.local/pipx/venvs/${DASHED_PACKAGE}${DASHED_VERSION}/lib/python3.10/site-packages/${PACKAGE}*"
    echo "${RM_CMD}"
    eval "${RM_CMD}"

    COMMAND="pipx inject ${PACKAGE}${VERSION} \"/opt/workspace/${PACKAGE}/dist/${PACKAGE}-${UPDATE_WITH_VERSION}-py3-none-any.whl\""
    echo "${COMMAND}"
    eval "${COMMAND}"
}

__gim_install_as_cli_tool() {
    # NOTE: uninstalling works but all dependencies are removed.
    pipx install dist/${PACKAGE}-${VERSION}-py3-none-any.whl --suffix ${VERSION}
    echo "Installed ${PACKAGE}-${VERSION} as a CLI tool"

    # From Copilot:
    # ${!injected_packages[@]} gives the indices of the array.
    # For each index, you get the package name and version, and construct the wheel path.
    for i in "${!injected_packages[@]}"; do
      pkg="${injected_packages[$i]}"
      ver="${injected_versions[$i]}"
      pipx inject "${PACKAGE}${VERSION}" "/opt/workspace/${pkg}/dist/${pkg}-${ver}-py3-none-any.whl"
    done
}

__gim_install() {
    # Install the package as a CLI tool
    # or as an importable package.
    if [ "${INSTALL_TYPE}" = "cli_tool" ]; then
        __gim_install_as_cli_tool
    elif [ "${INSTALL_TYPE}" = "importable_package" ]; then
        __gim_install_as_importable_package
    else
        echo "Unknown install type: ${INSTALL_TYPE}"
        return 1
    fi
}

_gim_update() {
    # Update the package as a CLI tool
    if [ "${INSTALL_TYPE}" = "cli_tool" ]; then
        __gim_update_cli_tool
    else
        echo "Unknown update type: ${INSTALL_TYPE}"
        return 1
    fi
}

if [ "${ACTION}" = "install" ]; then
    __gim_install
elif [ "${ACTION}" = "update" ]; then
    _gim_update
else
    echo "Unknown action: ${ACTION}"
    echo "Usage: deploy.sh <install|update> <version> [update_with_version]"
    return 1
fi
