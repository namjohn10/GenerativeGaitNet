//
// Created by lasagnaphil on 20. 9. 22..
//

#include "ShapeRenderer.h"

#include <glad/glad.h>
#include "GLfunctions.h"

std::vector<Eigen::Vector3f> buildCircle(float radius, int steps)
{
    std::vector<Eigen::Vector3f> points(steps + 1);
    if (steps < 2)
        return points;

    const float PI2 = acos(-1) * 2.0f;
    float x, y, a;
    for (int i = 0; i <= steps; ++i)
    {
        a = PI2 / steps * i;
        x = radius * cosf(a);
        y = radius * sinf(a);
        points[i] = Eigen::Vector3f(x, y, 0);
    }
    return points;
}

void transformContour(std::vector<Eigen::Vector3f> &contour, std::vector<Eigen::Vector3f> &normals,
                      const Eigen::Vector3f &pos, const Eigen::Vector3f &dir)
{
    Eigen::Quaternionf rot = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(0, 0, 1), dir);
    for (int i = 0; i < contour.size(); i++)
    {
        Eigen::Vector3f r = rot * contour[i];
        contour[i] = pos + r;
        normals[i] = r.normalized();
    }
}

void ShapeRenderer::renderMuscle(const MASS::Muscle *muscle)
{
    const float radius = 0.005f * sqrt(muscle->f0 / 1000.0);
    const int steps = 16;

    int N = muscle->GetAnchors().size();
    int numVertices = (steps + 1) * N;
    int numIndices = 6 * steps * (N - 1) + 2 * 3 * (steps - 1);

    std::vector<Eigen::Vector3f> waypoints(N);
    for (int i = 0; i < N; i++)
    {
        waypoints[i] = muscle->GetAnchors()[i]->GetPoint().cast<float>();
    }

    std::vector<std::vector<Eigen::Vector3f>> contours(N, buildCircle(radius, steps));
    std::vector<std::vector<Eigen::Vector3f>> contourNormals(N, std::vector<Eigen::Vector3f>(steps + 1));

    transformContour(contours[0], contourNormals[0], waypoints[0], (waypoints[1] - waypoints[0]).normalized());
    for (int i = 1; i < N - 1; i++)
    {
        Eigen::Vector3f dir1 = (waypoints[i] - waypoints[i - 1]).normalized();
        Eigen::Vector3f dir2 = (waypoints[i + 1] - waypoints[i]).normalized();
        Eigen::Vector3f dir = (dir1 + dir2).normalized();
        transformContour(contours[i], contourNormals[i], waypoints[i], dir);
    }
    transformContour(contours[N - 1], contourNormals[N - 1], waypoints[N - 1], (waypoints[N - 1] - waypoints[N - 2]).normalized());

    auto it = muscleVboIbo.find(muscle);
    if (it == muscleVboIbo.end())
    {
        GLuint vbo, ibo;
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ibo);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, 2 * 3 * sizeof(float) * numVertices, 0, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        std::vector<uint32_t> indices;
        for (int i = 0; i < N - 1; i++)
        {
            int k1 = i * (steps + 1);
            int k2 = (i + 1) * (steps + 1);
            for (int c = 0; c < steps; ++c, ++k1, ++k2)
            {
                indices.push_back(k1);
                indices.push_back(k1 + 1);
                indices.push_back(k2);

                indices.push_back(k2);
                indices.push_back(k1 + 1);
                indices.push_back(k2 + 1);
            }
        }

        int baseStartIdx = 0;
        int baseEndIdx = (steps + 1) * (N - 1);

        for (int i = 1; i < steps; ++i)
        {
            indices.push_back(baseStartIdx);
            indices.push_back(baseStartIdx + i);
            indices.push_back(baseStartIdx + i + 1);
        }
        for (int i = 1; i < steps; ++i)
        {
            indices.push_back(baseEndIdx);
            indices.push_back(baseEndIdx + i);
            indices.push_back(baseEndIdx + i + 1);
        }

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, numIndices * sizeof(uint32_t), indices.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        muscleVboIbo[muscle] = {vbo, ibo};
    }

    auto [vbo, ibo] = muscleVboIbo[muscle];

    std::vector<Eigen::Vector3f> vertices;
    std::vector<Eigen::Vector3f> normals;
    for (auto &section : contours)
    {
        vertices.insert(vertices.end(), section.begin(), section.end());
    }
    for (auto &section : contourNormals)
    {
        normals.insert(normals.end(), section.begin(), section.end());
    }

    assert(numVertices == vertices.size());

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, 3 * sizeof(float) * numVertices, vertices.data());
    glBufferSubData(GL_ARRAY_BUFFER, 3 * sizeof(float) * numVertices, 2 * 3 * sizeof(float) * numVertices, contourNormals.data());

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);

    glVertexPointer(3, GL_FLOAT, 0, (void *)0);
    glNormalPointer(GL_FLOAT, 0, (void *)(3 * sizeof(float) * numVertices));

    glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, (void *)0);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void ShapeRenderer::renderMesh(const MeshShape *meshShape, bool drawShadows, float shadowY,
                               const Eigen::Vector4d &color)
{

    const aiScene *scene = meshShape->getMesh();
    auto it = meshShapeVbo.find(meshShape);
    if (it == meshShapeVbo.end())
    {
        std::vector<uint32_t> vbo;
        createMeshVboRecursive(scene, vbo, scene->mRootNode);
        meshShapeVbo[meshShape] = vbo;
    }

    Eigen::Vector3d scale = meshShape->getScale();

    glColor4f(color[0], color[1], color[2], color[3]);
    glPushMatrix();
    glScaled(scale[0], scale[1], scale[2]);
    glPushAttrib(GL_POLYGON_BIT | GL_LIGHTING_BIT);

    auto vbo = meshShapeVbo[meshShape];
    int vboIdx = 0;
    renderMeshRecursive(scene, vbo, scene->mRootNode, vboIdx);

    glPopAttrib();
    glPopMatrix();
}

void ShapeRenderer::createMeshVboRecursive(const aiScene *scene, std::vector<uint32_t> &vbo,
                                           const aiNode *node)
{

    std::vector<float> buffer;

    for (int m = 0; m < node->mNumMeshes; m++)
    {
        GLuint m_vbo;
        glGenBuffers(1, &m_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
        const aiMesh *mesh = scene->mMeshes[node->mMeshes[m]];
        for (int t = 0; t < mesh->mNumFaces; ++t)
        {
            aiFace face = mesh->mFaces[t];
            for (int i = 0; i < face.mNumIndices; i++)
            {
                int index = face.mIndices[i];
                buffer.push_back(mesh->mVertices[index][0]);
                buffer.push_back(mesh->mVertices[index][1]);
                buffer.push_back(mesh->mVertices[index][2]);
                if (mesh->mNormals != nullptr)
                {
                    buffer.push_back(mesh->mNormals[index][0]);
                    buffer.push_back(mesh->mNormals[index][1]);
                    buffer.push_back(mesh->mNormals[index][2]);
                }
                if (mesh->mColors[0] != nullptr)
                {
                    buffer.push_back(mesh->mColors[0][index][0]);
                    buffer.push_back(mesh->mColors[0][index][1]);
                    buffer.push_back(mesh->mColors[0][index][2]);
                    buffer.push_back(mesh->mColors[0][index][3]);
                }
            }
        }
        glBufferData(GL_ARRAY_BUFFER, buffer.size() * sizeof(float), buffer.data(), GL_STATIC_DRAW);
        buffer.clear();
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        vbo.push_back(m_vbo);
    }
    for (int c = 0; c < node->mNumChildren; c++)
    {
        createMeshVboRecursive(scene, vbo, node->mChildren[c]);
    }
}

void ShapeRenderer::renderMeshRecursive(const aiScene *scene, const std::vector<uint32_t> &vbo,
                                        const aiNode *node, int &vboIdx)
{

    aiMatrix4x4 meshTrans = node->mTransformation;
    aiTransposeMatrix4(&meshTrans);
    glPushMatrix();
    glMultMatrixf((float *)&meshTrans);

    for (int m = 0; m < node->mNumMeshes; m++)
    {
        const aiMesh *mesh = scene->mMeshes[node->mMeshes[m]];

        bool enableNormals = mesh->mNormals != nullptr;
        bool enableColors = mesh->mColors[0] != nullptr;

        glBindBuffer(GL_ARRAY_BUFFER, vbo[vboIdx]);

        glEnableClientState(GL_VERTEX_ARRAY);
        if (enableNormals)
            glEnableClientState(GL_NORMAL_ARRAY);
        if (enableColors)
            glEnableClientState(GL_COLOR_ARRAY);

        int stride = 3 * sizeof(float);
        if (enableNormals)
            stride += 3 * sizeof(float);
        if (enableColors)
            stride += 4 * sizeof(float);
        glVertexPointer(3, GL_FLOAT, stride, (void *)0);
        if (enableNormals)
            glNormalPointer(GL_FLOAT, stride, (void *)(3 * sizeof(float)));
        if (enableColors)
            glColorPointer(4, GL_FLOAT, stride, (void *)(6 * sizeof(float)));

        // Assume that each face in mesh has the same vertices
        GLenum face_mode;
        switch (mesh->mFaces[0].mNumIndices)
        {
        case 1:
            face_mode = GL_POINTS;
            break;
        case 2:
            face_mode = GL_LINES;
            break;
        case 3:
            face_mode = GL_TRIANGLES;
            break;
        default:
            face_mode = GL_POLYGON;
            break;
        }

        glDrawArrays(face_mode, 0, mesh->mNumFaces * mesh->mFaces[0].mNumIndices);

        // TODO: draw shadows

        glDisableClientState(GL_VERTEX_ARRAY);
        if (enableNormals)
            glDisableClientState(GL_NORMAL_ARRAY);
        if (enableColors)
            glDisableClientState(GL_COLOR_ARRAY);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        vboIdx++;
    }

    for (int c = 0; c < node->mNumChildren; c++)
    {
        renderMeshRecursive(scene, vbo, node->mChildren[c], vboIdx);
    }

    glPopMatrix();
}
