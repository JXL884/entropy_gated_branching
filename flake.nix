{
  description = "Nix flake for dev shell with python & cuda.";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
    nixpkgs-unstable.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, nixpkgs-unstable }:
    let
      system = "x86_64-linux";
      nixpkgsConfig = {
        config = {
          allowUnfree = true;
          allowUnfreePredicate = (_: true);
          permittedInsecurePackages = [
            "electron-25.9.0"
            "qtwebkit-5.212.0-alpha4"
            "openssl-1.1.1w"
          ];
        };
      };
      pkgs = import nixpkgs ({
        inherit system;
        inherit (nixpkgsConfig) config;
      });
      unstable = import nixpkgs-unstable ({
        inherit system;
        inherit (nixpkgsConfig) config;
      });
      python = pkgs.python312.withPackages (ps: with ps; [
        setuptools
        wheel
        pkgs.catppuccin
        pygments
        pyqt6
        pip
        uv
      ]);
      deps = with pkgs; [
        clang
        llvmPackages_16.bintools
        rustup
        linuxPackages.nvidia_x11
        cudatoolkit
        cudaPackages.cudnn
        freeglut
        zlib
        gcc
        gcc.cc.lib
        stdenv.cc.cc.lib
        stdenv.cc
        libGLU
        libGL
        glib
        pango
        fontconfig
        cacert
        nodejs_22
        python
        python312Packages.matplotlib
      ];

      ld-lib-path = pkgs.lib.makeLibraryPath deps;
      lib-path = pkgs.lib.makeLibraryPath [ pkgs.cudatoolkit ];
    in {
      packages.${system}.fhsEnvironment = pkgs.buildFHSUserEnv {
        name = "entropix";

        targetPkgs = pkgs: deps;

        # nix run
        runScript = "zsh";
        profile = ''
          export LD_LIBRARY_PATH=${ld-lib-path}:$LD_LIBRARY_PATH;
          export LIBRARY_PATH=${lib-path}:$LIBRARY_PATH;
          export CUDA_PATH=${pkgs.cudatoolkit};
          export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
        '';
      };

      # nix develop
      devShells.${system}.default = pkgs.mkShell {
        name = "entropix";
        buildInputs = deps;
        shellHook = ''
          export LD_LIBRARY_PATH=${ld-lib-path}:$LD_LIBRARY_PATH
          export LIBRARY_PATH=${lib-path}:$LIBRARY_PATH
          export CUDA_PATH=${pkgs.cudatoolkit}
          export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
        '';
      };

    };
}
