package Docker

import jetbrains.buildServer.configs.kotlin.BuildType
import jetbrains.buildServer.configs.kotlin.Project
import jetbrains.buildServer.configs.kotlin.PublishMode

object DockerProject  : Project({
    name = "Docker"

    buildType(DeployDockerImage)
})

object DeployDockerImage : BuildType({

    enablePersonalBuilds = false
    type = Type.DEPLOYMENT
    maxRunningBuilds = 1
    publishArtifacts = PublishMode.SUCCESSFUL

})